"""What is actually behind the -0.45 N*m passive-term shift under an applied payload wrench?"""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
MOTORS=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]; ARM=MOTORS[:5]
HZ=500; G=9.80665
for load_dep in (True, False):
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=HZ,dynamics_robot_id="FT-02",
        emit_ground_truth=True,load_dependent_friction=load_dep))
    b.connect(); m,d=b._model,b._data
    jaw=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"robot0_moving_jaw")
    dof=[int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{x}")]) for x in ARM]
    o=b.get_observation(); mid={x:float(o[f"{x}.pos"]) for x in MOTORS}
    mid["shoulder_lift"]=-60.0; mid["elbow_flex"]=-40.0
    for _ in range(500): b.send_action({f"{x}.pos":mid[x] for x in MOTORS})
    fl=[];pa=[];qdd=[];qvel=[];bias=[]
    for i in range(int(HZ*8)):
        t=i/HZ; on=2.0<=t<5.0
        d.xfrc_applied[:]=0.0
        if on: d.xfrc_applied[jaw,2]=-0.200*G
        mujoco.mj_forward(m,d)
        fl.append([m.dof_frictionloss[j] for j in dof]); pa.append([d.qfrc_passive[j] for j in dof])
        qdd.append([d.qacc[j] for j in dof]); qvel.append([d.qvel[j] for j in dof])
        bias.append([-d.qfrc_bias[j] for j in dof])
        s=8*np.sin(2*np.pi*0.4*t)
        b.send_action({f"{x}.pos":mid[x]+(s if x in("shoulder_lift","elbow_flex") else 0.0) for x in MOTORS})
    b.disconnect()
    L=np.array(pa)>=0  # placeholder
    on=np.array([2.0<=i/HZ<5.0 for i in range(int(HZ*8))])
    fl,pa,qdd,qvel,bias=map(np.array,(fl,pa,qdd,qvel,bias))
    print(f"\nload_dependent_friction={load_dep}")
    print(f"{'joint':15} {'frictloss off':>13} {'frictloss on':>13} {'dPassive':>10} {'dq_mean':>9} {'dqdot_mean':>11} {'dGravity':>10}")
    for k,x in enumerate(ARM):
        print(f"{x:15} {fl[~on,k].mean():13.4f} {fl[on,k].mean():13.4f} "
              f"{pa[on,k].mean()-pa[~on,k].mean():+10.3f} "
              f"{qvel[on,k].mean()-qvel[~on,k].mean():+9.3f} "
              f"{qdd[on,k].mean()-qdd[~on,k].mean():+11.3f} "
              f"{bias[on,k].mean()-bias[~on,k].mean():+10.3f}")
