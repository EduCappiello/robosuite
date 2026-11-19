"""Physically honest alternative to inflating gripper armature: armature 0 on ALL six
(the identified value), and fix the actual cause — the jaw pressed against its joint
limit — by softening that limit's solref/solimp instead of faking inertia."""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
def run(tag, solref, solimp, zero_gripper=True):
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,kp=100,kv=20.0,
        dynamics_robot_id="FT-02",emit_ground_truth=False, allow_zero_armature=True,
        zero_armature_joints=tuple(M) if zero_gripper else tuple(M[:5])))
    b.connect(); m,d=b._model,b._data
    gj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_gripper")
    if solref is not None:
        m.jnt_solref[gj][:]=solref; m.jnt_solimp[gj][:]=solimp
    mujoco.mj_forward(m,d)
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in M}
    mx=0.;nan=0
    for i in range(2000):
        b.send_action({f"{k}.pos":init[k] for k in M})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
    sag={k:round(float(np.rad2deg(d.qpos[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")])]))-init[k],2) for k in M}
    print(f"{tag:34} max|qacc|={mx:11.1f} nan={nan:4d} ncon={d.ncon} sag={sag}")
    b.disconnect()
run("arm6zero, default limit", None, None)
run("arm6zero, solref .02/.3", [0.02,0.3], [0.9,0.95,0.001,0.5,2.0])
run("arm6zero, solref .05/.5", [0.05,0.5], [0.9,0.99,0.0001,0.5,2.0])
run("arm6zero, solimp soft", [0.03,0.5], [0.0,1.0,0.0,0.0,0.0])
run("arm5zero (current, gripper .028)", None, None, zero_gripper=False)
