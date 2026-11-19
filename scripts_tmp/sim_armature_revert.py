import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for tag,zaj in [("armature 0 on 5 arm joints (current)",tuple(M[:5])),
                ("armature 0.028 all six (upstream)",None)]:
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,kp=100,kv=20.0,
        dynamics_robot_id="FT-02",emit_ground_truth=True,zero_armature_joints=zaj))
    b.connect(); m,d=b._model,b._data
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in M}
    qa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_shoulder_lift")])
    qe=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_elbow_flex")])
    hl=[];he=[];mx=0.;nan=0
    for i in range(3000):
        b.send_action({f"{k}.pos":init[k]+(10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in M})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
        if i>=1500: hl.append(np.rad2deg(d.qpos[qa])); he.append(np.rad2deg(d.qpos[qe]))
    print(f"{tag:38} max|qacc|={mx:9.1f} nan={nan} lift_halfspan={(max(hl)-min(hl))/2:5.2f}deg elbow_halfspan={(max(he)-min(he))/2:5.2f}deg (cmd 10)")
    b.disconnect()
