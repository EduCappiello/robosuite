"""Model-vs-model at the REAL arm's own (q, qdot). No controller, no tracking: qpos/qvel are
written directly and mj_forward evaluates the model terms. Any difference left is a
difference between the two dynamics models, nothing else."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.scripts.lerobot_teleoperate import make_default_processors
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import make_so_torque_pipeline_steps
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]; ARM=M[:5]
LOG="/home/ecappiell/Documents/dev/lerobot-1-tg/outputs/next_free/ft02_boxes/w200g.npz"
d=np.load(LOG); pos=np.asarray(d["pos_deg"],float); vel=np.asarray(d["vel_hw_deg_s"],float)
cur=np.asarray(d["current_raw"],float); ld=np.asarray(d["load_raw"],float)
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02",
    emit_ground_truth=True))
b.connect(); m,dat=b._model,b._data
qadr=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{x}")]) for x in M]
dadr=[int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{x}")]) for x in M]
pipe=make_default_processors()[2]
pipe.steps=list(pipe.steps)+make_so_torque_pipeline_steps(motor_names=M,robot_type="so101_follower",
    robot_id="FT-02",estimation_method="ekf")
SG={x:[] for x in ARM}; SF={x:[] for x in ARM}; SM={x:[] for x in ARM}
EG={x:[] for x in ARM}; EF={x:[] for x in ARM}; EM={x:[] for x in ARM}
for i in range(0,len(pos),7):
    for k,x in enumerate(M):
        dat.qpos[qadr[k]]=np.deg2rad(pos[i,k]); dat.qvel[dadr[k]]=np.deg2rad(vel[i,k])
    mujoco.mj_forward(m,dat)
    o=b.get_observation()
    for x in ARM:
        SG[x].append(float(o[f"ground_truth.model.{x}.tau_gravity"]))
        SF[x].append(float(o[f"ground_truth.model.{x}.tau_friction"]))
    eo=pipe({**{f"{x}.pos":pos[i,k] for k,x in enumerate(M)},
             **{f"{x}.vel_hw":vel[i,k] for k,x in enumerate(M)},
             **{f"{x}.current_raw":cur[i,k] for k,x in enumerate(M)},
             **{f"{x}.load_raw":ld[i,k] for k,x in enumerate(M)}})
    for x in ARM:
        EG[x].append(float(eo[f"{x}.model/tau_gravity"]))
        EF[x].append(float(eo[f"{x}.model/tau_friction"]))
        EM[x].append(float(eo[f"{x}.model/tau_motor"]))
b.disconnect()
print("mean |term| over the real w200g trajectory (same q, qdot; sim terms from mj_forward)")
print(f"{'joint':15} {'G_sim':>8} {'G_ekf':>8} {'dG':>8} | {'F_sim':>8} {'F_ekf':>8} {'dF':>8} "
      f"| {'tau_mot_real':>12}")
for x in ARM:
    gs,ge=np.mean(np.abs(SG[x])),np.mean(np.abs(EG[x]))
    fs,fe=np.mean(np.abs(SF[x])),np.mean(np.abs(EF[x]))
    tm=np.mean(np.abs(EM[x]))
    print(f"{x:15} {gs:8.3f} {ge:8.3f} {gs-ge:+8.3f} | {fs:8.3f} {fe:8.3f} {fs-fe:+8.3f} | {tm:12.3f}")
