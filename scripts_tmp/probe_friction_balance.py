"""Is the sim's friction LIMIT consistent with the real arm's measured torques?

Friction is a reaction, not a law: the solver supplies whatever is needed up to the
breakaway limit. The estimator's kappa*|G| is a constitutive prediction. To compare them
fairly, drive the sim with the REAL measured motor torque at the REAL (q, qdot) and read
what the solver actually applies.

  needed = tau_motor_real - G - C - M*qdd   (qdd taken as 0: static holds dominate)
  applied = min(|needed|, limit) * sign(needed)
If needed exceeds the limit, the sim slips and the limit is too small; if not, the sim can
hold exactly what the real arm held.
"""
import json, numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.scripts.lerobot_teleoperate import make_default_processors
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import make_so_torque_pipeline_steps
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]; ARM=M[:5]
import sys
LOG=sys.argv[1] if len(sys.argv)>1 else "/home/ecappiell/Documents/dev/lerobot-1-tg/outputs/next_free/ft02_boxes/w0g.npz"
d0=np.load(LOG)
pos=np.asarray(d0["pos_deg"],float); vel=np.asarray(d0["vel_hw_deg_s"],float)
cur=np.asarray(d0["current_raw"],float); ld=np.asarray(d0["load_raw"],float)
LIVE=json.load(open("/home/ecappiell/Documents/dev/lerobot-1-tg/scripts/next_free_ekf_live.json"))
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02",
    emit_ground_truth=True))
b.connect(); m,dat=b._model,b._data
jid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{x}") for x in M]
qa=[int(m.jnt_qposadr[j]) for j in jid]; da=[int(m.jnt_dofadr[j]) for j in jid]
pipe=make_default_processors()[2]
pipe.steps=list(pipe.steps)+make_so_torque_pipeline_steps(motor_names=M,robot_type="so101_follower",
    robot_id="FT-02",estimation_method="ekf")
res={x:[] for x in ARM}
for i in range(0,len(pos),7):
    eo=pipe({**{f"{x}.pos":pos[i,k] for k,x in enumerate(M)},
             **{f"{x}.vel_hw":vel[i,k] for k,x in enumerate(M)},
             **{f"{x}.current_raw":cur[i,k] for k,x in enumerate(M)},
             **{f"{x}.load_raw":ld[i,k] for k,x in enumerate(M)}})
    for k,x in enumerate(M):
        dat.qpos[qa[k]]=np.deg2rad(pos[i,k]); dat.qvel[da[k]]=np.deg2rad(vel[i,k])
    dat.qfrc_applied[:]=0.0
    for k,x in enumerate(ARM):
        dat.qfrc_applied[da[k]]=float(eo[f"{x}.model/tau_motor"])   # inject the REAL torque
    dat.qacc[:]=0.0
    mujoco.mj_forward(m,dat)
    nefc=int(dat.nefc); rows=[r for r in range(nefc) if dat.efc_type[r]==1]
    applied={int(dat.efc_id[r]):float(dat.efc_force[r]) for r in rows}
    for k,x in enumerate(ARM):
        # AUDITED version. The first cut used needed = tau_motor - G, which silently charged
        # Coriolis and inertia to friction and inflated the demand. Full balance:
        #   needed = tau_motor - G - C - M*qdd
        G=float(-dat.qfrc_bias[da[k]]); tm=float(eo[f"{x}.model/tau_motor"])
        C=float(eo[f"{x}.model/tau_coriolis"]); I=float(eo[f"{x}.model/tau_inertial"])
        needed=tm-G-C-I
        lim=float(m.dof_frictionloss[da[k]])
        # The limit the FIX intends, computed here so the probe does not have to depend on
        # the driver's control loop having run (it has not, in this path).
        pj=LIVE["per_joint"][x]
        Gmag=abs(float(-dat.qfrc_bias[da[k]]))
        lim_fix=float(pj["coulomb_Fc"])+float(pj.get("kappa_static",pj["kappa"]))*Gmag
        res[x].append((needed, applied.get(da[k],0.0), lim, float(eo[f"{x}.model/tau_friction"]),
                       tm, G, C, I, lim_fix))
b.disconnect()
print("LOG:", LOG)
print("needed = tau_motor - G - C - M*qdd  (full balance; the first cut omitted C and I)")
print(f"{'joint':15} {'need|F|':>9} {'limit':>8} {'ekf law':>9} {'slip%':>7} | "
      f"{'mean|tau_m|':>11} {'mean|G|':>9} {'mean|C|':>9} {'mean|I|':>9}")
for x in ARM:
    a=np.array(res[x]); need, gave, lim, law = a[:,0], a[:,1], a[:,2], a[:,3]
    tm,G,C,I = a[:,4], a[:,5], a[:,6], a[:,7]
    slip=float(np.mean(np.abs(need)>lim*1.02)*100)
    lim_fix=a[:,8]
    slip_fix=float(np.mean(np.abs(need)>lim_fix*1.02)*100)
    print(f"{x:15} {np.mean(np.abs(need)):9.3f} {np.mean(lim):8.3f} {np.mean(np.abs(law)):9.3f} "
          f"{slip:7.1f} | {np.mean(np.abs(tm)):11.3f} {np.mean(np.abs(G)):9.3f} "
          f"{np.mean(np.abs(C)):9.3f} {np.mean(np.abs(I)):9.3f}  "
          f"fix_limit={np.mean(lim_fix):.3f} slip_fix={slip_fix:5.1f}%")
print(f"\ncurrent-register quantization floor: ~0.071 N*m per sample (measured). "
      f"Any |needed| below that is indistinguishable from zero.")
