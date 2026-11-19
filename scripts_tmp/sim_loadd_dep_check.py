import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
MOTORS=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
def raw(m): return m._model if hasattr(m,"_model") else m
cfg=RobosuiteSimConfig(id="sim",control_freq=500,dynamics_robot_id="FT-02",emit_ground_truth=True)
bot=RobosuiteSimFollower(cfg); bot.connect()
m,d=bot._model,bot._data
jaw=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"robot0_moving_jaw"); base=float(m.body_mass[jaw])
def dof(n):
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_{n}"); return int(m.jnt_dofadr[j])
obs=bot.get_observation(); init={k:float(obs[f"{k}.pos"]) for k in MOTORS}
def hold(n):
    for _ in range(n): bot.send_action({f"{k}.pos":init[k] for k in MOTORS})

print("frictionloss after configure:", [round(float(m.dof_frictionloss[dof(k)]),4) for k in MOTORS])
hold(300)
fl_empty=[float(m.dof_frictionloss[dof(k)]) for k in MOTORS]
print("frictionloss @empty  :", [round(v,4) for v in fl_empty])
for g in (50,200):
    m.body_mass[jaw]=base+g/1000.0; mujoco.mj_forward(m,d); hold(300)
    fl=[float(m.dof_frictionloss[dof(k)]) for k in MOTORS]
    print(f"frictionloss @{g}g  :", [round(v,4) for v in fl], "delta:", [round(a-b,4) for a,b in zip(fl,fl_empty)])
m.body_mass[jaw]=base; mujoco.mj_forward(m,d)
mx=0;nan=0
for i in range(2000):
    bot.send_action({f"{k}.pos": init[k]+ (10*np.sin(2*np.pi*0.5*i/500) if k in("shoulder_lift","elbow_flex") else 0) for k in MOTORS})
    if not np.isfinite(d.qacc).all(): nan+=1
    else: mx=max(mx,float(np.max(np.abs(d.qacc))))
print(f"stability 2000 steps: max|qacc|={mx:.1f} nan={nan}")
bot.disconnect()
print("OK" if nan==0 and mx<2000 else "BAD")
