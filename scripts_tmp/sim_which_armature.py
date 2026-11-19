"""Which armature actually matters? Zero subsets and see. Hypothesis: the gripper jaw
(inertia 1.6e-5, parked against its joint limit = an active MuJoCo constraint) is what
breaks at 1000 Hz, not the arm joints."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
MOTORS=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
ARM=MOTORS[:5]
SUBSETS={"none (XML 0.028 kept)":[], "gripper only":["gripper"], "wrist_roll only":["wrist_roll"],
         "wrist_roll+gripper":["wrist_roll","gripper"], "wrist_flex+roll+grip":["wrist_flex","wrist_roll","gripper"],
         "all 5 arm joints":ARM, "all 6":MOTORS}
for tag,zero in SUBSETS.items():
    cfg=RobosuiteSimConfig(id="sim",control_freq=500,kp=100,kv=20.0,dynamics_robot_id="FT-02",emit_ground_truth=False)
    b=RobosuiteSimFollower(cfg); b.connect()
    m,d=b._model,b._data
    dof={k:int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")]) for k in MOTORS}
    for k in zero: m.dof_armature[dof[k]]=0.0
    mujoco.mj_forward(m,d)
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in MOTORS}
    mx=0.;nan=0
    for i in range(1000):
        b.send_action({f"{k}.pos":init[k] for k in MOTORS})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
    sag=np.array([float(np.rad2deg(d.qpos[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")])]))-init[k] for k in MOTORS])
    print(f"{tag:26} -> max|qacc|={mx:12.1f} nan={nan:4d} sag_deg={np.round(sag,2)} ncon={d.ncon}")
    b.disconnect()
