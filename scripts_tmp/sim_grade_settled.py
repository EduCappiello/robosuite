#!/usr/bin/env python
"""Settled-window grading: score the EKF against tau_ext_residual with the transient excluded.

sim_grade_ablate.py showed the payload "overestimate" was mostly viscous damping resisting the
arm's sag to a new equilibrium, and that tau_ext_jac vs tau_ext_residual differ by up to 1.49 N*m.
This run removes both confounds:
  * the payload wrench is RAMPED in over RAMP_S, so no step discontinuity;
  * scoring uses only settled windows (baseline before the ramp, tail after the transient);
  * the target is tau_ext_residual -- the same definition the estimator computes.

Run:
  cd ~/Documents/dev/robosuite
  PYTHONPATH=$PWD:~/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_grade_settled.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import (
    make_so_torque_pipeline_steps,
)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]
HZ = 500
SECONDS = 12.0
T_ON, RAMP_S, T_OFF = 3.0, 1.5, 9.0
PAYLOAD_KG = 0.200
G = 9.80665
SETTLE_S = 2.0          # discard this much of the window after each load edge


def collect():
    bot = RobosuiteSimFollower(RobosuiteSimConfig(
        id="sim", control_freq=HZ, dynamics_robot_id="FT-02", emit_ground_truth=True))
    bot.connect()
    model, data = bot._model, bot._data
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")

    obs = bot.get_observation()
    mid = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    mid["shoulder_lift"] = -60.0
    mid["elbow_flex"] = -40.0
    for _ in range(500):
        bot.send_action({f"{m}.pos": mid[m] for m in MOTORS})

    n = int(HZ * SECONDS)
    parity, res, jac, qvel = [], [], [], []
    arm_dof = [int(model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")]) for m in ARM]
    for i in range(n):
        t = i / HZ
        data.xfrc_applied[:] = 0.0
        if T_ON <= t < T_ON + RAMP_S:
            frac = (t - T_ON) / RAMP_S
        elif T_ON + RAMP_S <= t < T_OFF:
            frac = 1.0
        else:
            frac = 0.0
        data.xfrc_applied[jaw, 2] = -PAYLOAD_KG * G * frac
        mujoco.mj_forward(model, data)

        obs = bot.get_observation()
        parity.append({k: v for k, v in obs.items() if not k.startswith("ground_truth")})
        res.append({m: float(obs[f"ground_truth.ext.{m}.tau_ext_residual"]) for m in MOTORS})
        jac.append({m: float(obs[f"ground_truth.ext.{m}.tau_ext_jac"]) for m in MOTORS})
        qvel.append([float(data.qvel[d]) for d in arm_dof])

        # Hold the pose fixed: this is a load test, not a tracking test.
        bot.send_action({f"{m}.pos": mid[m] for m in MOTORS})
    bot.disconnect()
    return parity, res, jac, np.array(qvel)


def run_estimator(parity, method):
    from lerobot.scripts.lerobot_teleoperate import make_default_processors

    pipeline = make_default_processors()[2]
    pipeline.steps = list(pipeline.steps) + make_so_torque_pipeline_steps(
        motor_names=MOTORS, robot_type="so101_follower", robot_id="FT-02",
        estimation_method=method)
    out = []
    for obs in parity:
        o = pipeline(dict(obs))
        out.append({m: float(o.get(f"{m}.estimated/tau_ext", np.nan)) for m in MOTORS})
    return out


def windows():
    t = np.arange(int(HZ * SECONDS)) / HZ
    base = (t >= 0.5) & (t < T_ON)
    load = (t >= T_ON + RAMP_S + SETTLE_S) & (t < T_OFF)
    return base, load


def main() -> None:
    parity, res, jac, qvel = collect()
    base, load = windows()
    print(f"settled windows: baseline {base.sum()/HZ:.1f}s, loaded {load.sum()/HZ:.1f}s "
          f"(ramp {RAMP_S}s, settle {SETTLE_S}s discarded); pose held fixed")
    for method in ("ekf", "lpf"):
        est = run_estimator(parity, method)
        print(f"\n=== {method.upper()} vs tau_ext_residual, settled, static pose ===")
        print(f"{'joint':15} {'gt res step':>12} {'est step':>10} {'err':>8} | "
              f"{'gt jac step':>11} | {'corr':>6} {'RMSE':>7} {'dq step':>8}")
        for k, m in enumerate(ARM):
            r = np.array([x[m] for x in res])
            j = np.array([x[m] for x in jac])
            e = np.array([x[m] for x in est])
            st_r = r[load].mean() - r[base].mean()
            st_j = j[load].mean() - j[base].mean()
            st_e = e[load].mean() - e[base].mean()
            mid = load | base
            corr = float(np.corrcoef(r[mid], e[mid])[0, 1])
            rmse = float(np.sqrt(np.mean((e[mid] - r[mid]) ** 2)))
            dq = qvel[load, k].mean() - qvel[base, k].mean()
            print(f"{m:15} {st_r:+12.3f} {st_e:+10.3f} {st_e-st_r:+8.3f} | "
                  f"{st_j:+11.3f} | {corr:6.3f} {rmse:7.3f} {dq:+8.4f}")


if __name__ == "__main__":
    main()
