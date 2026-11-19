#!/usr/bin/env python
"""Time-series audit of the shoulder_lift sine: is the poor tracking a joint-limit effect
(init qpos -104.13 deg is OUTSIDE the XML range [-100, +100]) or a gain effect?

Simulation only.

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_lift_timeseries.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
KV = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
CENTER = float(sys.argv[3]) if len(sys.argv) > 3 else -104.129


def raw_model(m):
    while not isinstance(m, mujoco.MjModel):
        m = m._model
    return m


def raw_data(d):
    while not isinstance(d, mujoco.MjData):
        d = d._data
    return d


def main() -> int:
    cfg = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="FT-02",
                             emit_ground_truth=True, has_renderer=False,
                             friction_overrides={m: {"armature": 0.0} for m in MOTORS[:5]},
                             kp=KP, kv=KV)
    bot = RobosuiteSimFollower(cfg)
    bot.connect()
    env = bot.sim.env
    model, data = raw_model(env.sim.model), raw_data(env.sim.data)
    ctrl = env.robots[0].composite_controller.part_controllers["right"]
    ctrl.kp = np.full(ctrl.control_dim, KP)
    ctrl.kd = np.full(ctrl.control_dim, KV)

    obs = bot.get_observation()
    init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    init_deg["shoulder_lift"] = CENTER
    for _ in range(500):
        bot.send_action({f"{m}.pos": init_deg[m] for m in MOTORS})

    N = 2000
    q = np.zeros(N)
    c = np.zeros(N)
    tau = np.zeros(N)
    lam = np.zeros(N)
    qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_shoulder_lift")])
    dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_shoulder_lift")])
    for i in range(N):
        tgt = dict(init_deg)
        tgt["shoulder_lift"] = CENTER + 10.0 * np.sin(2 * np.pi * 0.5 * i / 500.0)
        bot.send_action({f"{m}.pos": tgt[m] for m in MOTORS})
        q[i] = np.rad2deg(data.qpos[qadr])
        c[i] = tgt["shoulder_lift"]
        tau[i] = data.qfrc_actuator[dof]
        lam[i] = data.qfrc_constraint[dof]
    print(f"kp={KP} kv={KV} center={CENTER}  joint range = "
          f"{np.rad2deg(model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot0_shoulder_lift')])}")
    print(f"cmd  min/max = {c.min():+.3f} / {c.max():+.3f} deg")
    print(f"qpos min/max = {q.min():+.3f} / {q.max():+.3f} deg   span_pp={q.max() - q.min():.3f}")
    print(f"tau  min/max = {tau.min():+.4f} / {tau.max():+.4f} N.m")
    print(f"lim  min/max = {lam.min():+.4f} / {lam.max():+.4f} N.m (joint-limit constraint force)")
    print("\n  i    cmd      qpos     err     tau      lam")
    for i in range(400, 1401, 50):
        print(f"{i:5d} {c[i]:+8.3f} {q[i]:+8.3f} {c[i] - q[i]:+7.3f} {tau[i]:+8.4f} {lam[i]:+8.4f}")
    bot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
