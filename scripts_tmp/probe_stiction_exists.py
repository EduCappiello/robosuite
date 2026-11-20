"""Does MuJoCo's dof_frictionloss act as a STATIC breakaway limit, or only when moving?

This decides whether Problem 1 is real. The gt term is -(sign(v)*frictionloss + damping*v),
which is 0 at v=0 by definition -- so reading it at a static hold proves nothing. Here we
apply NO actuator torque and step the raw model: if the joint stays put, frictionloss is
holding it, i.e. MuJoCo already gives us stiction and the fix is to set the right VALUE.
If it slides, frictionloss is kinetic-only and we need a real static-friction element.
"""
import numpy as np, mujoco
from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

b = RobosuiteSimFollower(RobosuiteSimConfig(id="s", control_freq=500, dynamics_robot_id="FT-02"))
b.connect()
m, d = b._model, b._data
lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_shoulder_lift")
qa, da = int(m.jnt_qposadr[lift]), int(m.jnt_dofadr[lift])

# Settle the whole arm at a mid-range pose once, then restore that exact qpos for every
# trial. Restoring only the lift joint let the other joints carry over each trial's drift,
# so the gravity torque being held changed from row to row.
for _ in range(1000):
    b.send_action({f"{x}.pos": v for x, v in
                   zip(["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
                       [0.0, -60.0, -40.0, 20.0, 0.0, 0.0])})
q_ref = d.qpos.copy()

for fl in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00):
    m.dof_frictionloss[da] = fl
    m.dof_damping[da] = 0.0
    m.dof_armature[da] = 0.028
    d.qpos[:] = q_ref
    d.qvel[:] = 0.0
    d.qfrc_applied[:] = 0.0
    d.qacc[:] = 0.0
    d.ctrl[:] = 0.0                      # no actuator at all
    mujoco.mj_forward(m, d)
    g = float(-d.qfrc_bias[da])          # gravity torque the friction must hold
    q0 = float(d.qpos[qa])
    for _ in range(500):
        mujoco.mj_step(m, d)
    moved = float(np.rad2deg(d.qpos[qa] - q0))
    print(f"frictionloss={fl:5.2f}  gravity torque={g:+6.3f} N*m  "
          f"drift over 0.5 s with ZERO actuator: {moved:+8.4f} deg  "
          f"-> {'HOLDS (static limit)' if abs(moved) < 0.05 else 'SLIDES (kinetic only)'}")
b.disconnect()
