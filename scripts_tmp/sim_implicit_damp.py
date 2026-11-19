"""(c) probe: armature=0 is only stabilisable if the velocity feedback is IMPLICIT.
MuJoCo integrates dof_damping implicitly (unconditionally stable); the controller's kv
enters as an explicit applied torque (limit kv < 2M/dt). So: kv_ctrl=0, damping in the
model. Does armature=0 then hold at 1000 Hz?"""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
MOTORS=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for kp,kd_model in [(100.,0.6),(100.,1.0),(150.,1.0),(50.,0.5),(100.,0.0)]:
    cfg=RobosuiteSimConfig(id="sim",control_freq=500,kp=kp,kv=0.0,dynamics_robot_id="FT-02",
                           emit_ground_truth=False)
    b=RobosuiteSimFollower(cfg); b.connect()
    m,d=b._model,b._data
    dof={k:int(m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")]) for k in MOTORS}
    for k in MOTORS:
        m.dof_armature[dof[k]]=0.0
        m.dof_damping[dof[k]]=kd_model
    mujoco.mj_forward(m,d)
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in MOTORS}
    mx=0.;nan=0
    for i in range(2000):
        b.send_action({f"{k}.pos": init[k]+(10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in MOTORS})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
    q=env_q=np.array([float(np.rad2deg(d.qpos[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{k}")])])) for k in MOTORS])
    print(f"kp={kp:<5g} kv_ctrl=0 dof_damping={kd_model:<4g} armature=0 -> max|qacc|={mx:10.1f} nan={nan} "
          f"sag_deg={np.round(q-np.array([init[k] for k in MOTORS]),2)}")
    b.disconnect()
