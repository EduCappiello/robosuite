"""Is the sim's tau_measured (from current_raw) on the same scale as the real arm's?
NEXT computes tau_ext = tau_measured - tau_free_learned. If the sim's current registers
imply a different torque scale, NEXT's learned tau_free is in the wrong units for the
input and tau_ext is meaningless -- a model difference that hits NEXT but not the EKF."""
import numpy as np, glob, os
from lerobot.robots.so_follower.dynamics.motor_parameters import (
    compute_motor_torque_signed, SO101_FOLLOWER_GEAR_RATIOS)
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
print(f"{'source':28} {'n':>6} " + " ".join(f"{m[:9]:>9}" for m in M))
for path in sorted(glob.glob(os.path.expanduser(
        "~/Documents/dev/lerobot-1-tg/outputs/next_free/ft02_boxes/w*g.npz"))):
    d=np.load(path)
    cur=np.asarray(d['current_raw'],float); ld=np.asarray(d['load_raw'],float)
    vel=np.asarray(d['vel_hw_deg_s'],float)/57.2958
    t=np.array([compute_motor_torque_signed(i_motor_raw=cur[i], load_raw=ld[i], dq=vel[i],
              gear_ratios=SO101_FOLLOWER_GEAR_RATIOS, motor_names=M) for i in range(len(cur))])
    name=os.path.basename(path)[:-4]
    print(f"REAL {name:22} {len(t):6d} " + " ".join(f"{np.median(np.abs(t[:,j])):9.3f}" for j in range(6)))

# --- the same quantity, produced by the sim's parity tier -------------------
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
import mujoco
b=RobosuiteSimFollower(RobosuiteSimConfig(id="s",control_freq=500,dynamics_robot_id="FT-02"))
b.connect()
o=b.get_observation(); mid={x:float(o[f"{x}.pos"]) for x in M}
mid["shoulder_lift"]=-60.0; mid["elbow_flex"]=-40.0
for _ in range(500): b.send_action({f"{x}.pos":mid[x] for x in M})
jaw=mujoco.mj_name2id(b._model,mujoco.mjtObj.mjOBJ_BODY,"robot0_moving_jaw")
rows=[]
for i in range(4000):
    t=i/500; b._data.xfrc_applied[:]=0.0
    if 2.0<=t<5.0: b._data.xfrc_applied[jaw,2]=-0.200*9.80665
    mujoco.mj_forward(b._model,b._data)
    o=b.get_observation()
    rows.append(([float(o[f"{x}.current_raw"]) for x in M],
                 [float(o[f"{x}.load_raw"]) for x in M],
                 [float(o[f"{x}.vel_hw"]) for x in M]))
    b.send_action({f"{x}.pos":mid[x] for x in M})
b.disconnect()
t=np.array([compute_motor_torque_signed(i_motor_raw=c,load_raw=l,dq=[v/57.2958 for v in vv],
      gear_ratios=SO101_FOLLOWER_GEAR_RATIOS,motor_names=M) for c,l,vv in rows])
print(f"SIM  held+0.2kg             {len(t):6d} " + " ".join(f"{np.median(np.abs(t[:,j])):9.3f}" for j in range(6)))
print(f"SIM  unique |tau| values    {len(np.unique(np.round(np.abs(t[:,1]),4)))} "
      f"(real w200g lift unique: {len(np.unique(np.round(np.abs(np.array([0.0])) ,4)))} -> see quantization above)")
