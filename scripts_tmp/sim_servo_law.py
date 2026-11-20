"""Compare the two actuator laws on identical motion: robosuite's M-scaled acceleration law
vs a Feetech-style torque-space PD with the datasheet stall clamp.
Both are driven from a mid-range pose with a +/-10 deg, 0.4 Hz sine on lift+elbow."""
import numpy as np
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
def run(**kw):
    b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02",**kw))
    b.connect(); o=b.get_observation()
    mid={x:float(o[f"{x}.pos"]) for x in M}; mid["shoulder_lift"]=-60.; mid["elbow_flex"]=-40.
    for i in range(500): b.send_action({f"{x}.pos":mid[x] for x in M})
    hs={x:[] for x in ("shoulder_lift","elbow_flex")}; mx=0.; nan=0; sat=0
    for i in range(2000):
        s=10*np.sin(2*np.pi*0.4*i/500)
        act={f"{x}.pos":mid[x]+(s if x in hs else 0.) for x in M}
        b.send_action(act)
        if not np.isfinite(b._data.qacc).all(): nan+=1
        else: mx=max(mx,float(np.max(np.abs(b._data.qacc))))
        if i>=1000:
            for x in hs:
                import mujoco as _mj
                j=b._model.jnt_qposadr[_mj.mj_name2id(b._model,_mj.mjtObj.mjOBJ_JOINT,f"robot0_{x}")]
                hs[x].append(float(np.rad2deg(b._data.qpos[j])))
        if getattr(b.config,"servo_law","")=="torque":
            lim=b.config.servo_torque_limit
            sat+=int(np.any(np.abs(b._data.ctrl[:5])>=lim*0.999))
    b.disconnect()
    return {x:round((max(v)-min(v))/2,2) for x,v in hs.items()}, mx, nan, sat

print(f"{'law':28} {'lift hs':>8} {'elbow hs':>9} {'max|qacc|':>10} {'nan':>4} {'sat steps':>10}")
base=dict(kp=2400,kv=50)
h,mx,nan,sat=run(**base)
print(f"{'m_scaled (kp2400/kv50)':28} {h['shoulder_lift']:8.2f} {h['elbow_flex']:9.2f} {mx:10.1f} {nan:4d} {sat:10d}")
for kp,kd in [(60,2),(120,4),(240,8),(480,14),(960,25)]:
    h,mx,nan,sat=run(servo_law="torque",servo_kp=kp,servo_kd=kd)
    print(f"{'torque kp=%d kd=%d'%(kp,kd):28} {h['shoulder_lift']:8.2f} {h['elbow_flex']:9.2f} "
          f"{mx:10.1f} {nan:4d} {sat:10d}")
