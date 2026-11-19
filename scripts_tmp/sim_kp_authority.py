"""armature=0 on the 5 arm joints is stable but the M-scaled PD loses torque authority.
Sweep kp (and kv) to find gains that both stay stable and actually track."""
import numpy as np, mujoco, time
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for kp in (100,300,600,1000,1500,2500):
  for kv in (5,20,50):
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,kp=kp,kv=kv,
        dynamics_robot_id="FT-02",emit_ground_truth=False))
    b.connect(); m,d=b._model,b._data
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in M}
    mx=0.;nan=0;hist=[]
    qa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_shoulder_lift")])
    t0=time.perf_counter()
    for i in range(2000):
        b.send_action({f"{k}.pos":init[k]+(10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in M})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
        if i>=1000: hist.append(float(np.rad2deg(d.qpos[qa])))
    wall=time.perf_counter()-t0
    span=(max(hist)-min(hist))/2 if hist else 0.
    print(f"kp={kp:<5g} kv={kv:<3g} max|qacc|={mx:12.1f} nan={nan:4d} lift_halfspan={span:6.2f} deg (cmd 10) rt={2000/500/wall:.2f}x")
    b.disconnect()
