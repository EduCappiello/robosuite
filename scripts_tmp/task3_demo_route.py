"""Task 3 route A: can the lerobot sim follower drive cupPnP_task2 on the desk arm at 500 Hz?
No hardware, no recording - connect, step, report the achievable rate and the new obs."""
import time, numpy as np
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
M=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
cfg=RobosuiteSimConfig(id="sim", robot="soarm101", env="cupPnP_task2", control_freq=500,
                       dynamics_robot_id="FT-02", emit_ground_truth=True,
                       has_renderer=False)
b=RobosuiteSimFollower(cfg); b.connect()
env=b.sim.env
print("env", type(env).__name__, "| physics Hz", round(1/float(env.model_timestep)),
      "| control Hz", env.control_freq, "| substeps", int(env.control_timestep/env.model_timestep))
print("action_dim", env.action_dim, "| success_mode", getattr(env,"success_mode",None))
o=b.get_observation()
print("obs keys:", len(o), "| parity pos:", sum(1 for k in o if k.endswith(".pos")),
      "| gt:", sum(1 for k in o if k.startswith("ground_truth")))
act={f"{k}.pos": float(o[f"{k}.pos"]) for k in M}
t0=time.perf_counter(); n=1000
for _ in range(n): b.send_action(act)
dt=time.perf_counter()-t0
print(f"{n} steps @500Hz wall={dt:.2f}s -> {n/500/dt:.2f}x realtime; can a 500 Hz logger keep up? {n/500/dt>=1.0}")
w=env.read_wrist_ft(); print("read_wrist_ft force", np.round(w["force"],3), "torque", np.round(w["torque"],3))
print("brew travel", env._brew_travel_m, "pressed", env._pressed)
b.disconnect(); print("ROUTE A OK")
