"""G5: the M-scaled PD gives the arm little authority. Before changing the controller law,
check whether retuning kp/kv inside the explicit-Euler ceiling is enough.
armature 0.028 -> M ~ 0.03 -> kp < 4M/dt^2 = 12000, kv < 2M/dt = 60."""
import numpy as np, mujoco, time
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
def run(kp,kv):
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,kp=kp,kv=kv,
        dynamics_robot_id="FT-02",emit_ground_truth=False))
    b.connect(); m,d=b._model,b._data
    o=b.get_observation(); init={k:float(o[f"{k}.pos"]) for k in M}
    tgt=dict(init); tgt["shoulder_lift"]=-60.0; tgt["elbow_flex"]=-40.0
    for i in range(500): b.send_action({f"{k}.pos":tgt[k] for k in M})
    qa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_shoulder_lift")])
    qe=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot0_elbow_flex")])
    hl=[];he=[];err=[];mx=0.;nan=0
    t0=time.perf_counter()
    for i in range(2000):
        s=10*np.sin(2*np.pi*0.5*i/500)
        b.send_action({f"{k}.pos":tgt[k]+(s if k in("shoulder_lift","elbow_flex") else 0) for k in M})
        if not np.isfinite(d.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(d.qacc))))
        if i>=1000:
            hl.append(np.rad2deg(d.qpos[qa])); he.append(np.rad2deg(d.qpos[qe]))
            err.append(abs((tgt["shoulder_lift"]+s)-np.rad2deg(d.qpos[qa])))
    wall=time.perf_counter()-t0
    hs=(max(hl)-min(hl))/2; he2=(max(he)-min(he))/2
    print(f"kp={kp:<6g} kv={kv:<4g} lift_halfspan={hs:5.2f} elbow_halfspan={he2:5.2f} (cmd 10) "
          f"rms_err={np.mean(err):5.2f}deg max|qacc|={mx:9.1f} nan={nan} rt={2000/500/wall:.2f}x")
    b.disconnect()
for kp,kv in [(100,20),(300,20),(600,30),(1200,40),(2400,50),(4000,55),(6000,58),(8000,59)]:
    run(kp,kv)
