#!/usr/bin/env python
"""Grade the real-robot torque estimators (EKF / NEXT) against MuJoCo ground truth.

The sim publishes two tiers on the same sample:
  parity tier      {motor}.pos / .vel_hw / .current_raw / .load_raw  -- exactly what the
                   real FT-02 bus gives the estimator
  privileged tier  ground_truth.model.*  and ground_truth.ext.*.tau_ext -- the exact answer

So the estimator is fed nothing it would not get on hardware, and scored against something
it could never get on hardware. That is the point: it separates estimator error from
model error.

Run:
  cd ~/Documents/dev/robosuite
  PYTHONPATH=$PWD:~/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_grade_estimators.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import (
    make_so_torque_pipeline_steps,
)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]
HZ = 500
SECONDS = 8.0
PAYLOAD_START_S, PAYLOAD_END_S = 2.0, 5.0
PAYLOAD_KG = 0.200


def collect():
    """Drive the sim through a sine sweep with a 200 g payload appearing mid-run."""
    bot = RobosuiteSimFollower(RobosuiteSimConfig(
        id="sim", control_freq=HZ, dynamics_robot_id="FT-02", emit_ground_truth=True))
    bot.connect()
    model, data = bot._model, bot._data
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")
    base_mass = float(model.body_mass[jaw])

    obs = bot.get_observation()
    init = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    n = int(HZ * SECONDS)
    parity, truth = [], []
    payload_windows = []
    for i in range(n):
        t = i / HZ
        loaded = PAYLOAD_START_S <= t < PAYLOAD_END_S
        model.body_mass[jaw] = base_mass + (PAYLOAD_KG if loaded else 0.0)
        mujoco.mj_forward(model, data)
        payload_windows.append(loaded)

        obs = bot.get_observation()
        parity.append({k: v for k, v in obs.items() if not k.startswith("ground_truth")})
        truth.append({m: float(obs[f"ground_truth.ext.{m}.tau_ext"]) for m in MOTORS})

        s = 8.0 * np.sin(2 * np.pi * 0.4 * t)
        goal = {f"{m}.pos": init[m] + (s if m in ("shoulder_lift", "elbow_flex") else 0.0)
                for m in MOTORS}
        bot.send_action(goal)
    bot.disconnect()
    return parity, truth, np.array(payload_windows)


def run_estimator(parity, method, robot_id="FT-02"):
    # Same wiring teleop uses: the steps are appended to the default robot-observation
    # pipeline, and the pipeline is called with a raw observation dict.
    from lerobot.scripts.lerobot_teleoperate import make_default_processors

    pipeline = make_default_processors()[2]
    pipeline.steps = list(pipeline.steps) + make_so_torque_pipeline_steps(
        motor_names=MOTORS, robot_type="so101_follower", robot_id=robot_id,
        estimation_method=method)
    out = []
    for obs in parity:
        o = pipeline(dict(obs))
        out.append({m: float(o.get(f"{m}.estimated/tau_ext", np.nan)) for m in MOTORS})
    return out


def score(truth, est, loaded):
    rows = []
    for m in ARM:
        t = np.array([r[m] for r in truth])
        e = np.array([r[m] for r in est])
        ok = np.isfinite(e)
        if not ok.any():
            rows.append((m, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        t2, e2, l2 = t[ok], e[ok], loaded[ok]
        free = ~l2
        # The payload step is the load-bearing test: does the estimator see the 200 g?
        step_truth = t2[l2].mean() - t2[free].mean()
        step_est = e2[l2].mean() - e2[free].mean()
        rows.append((
            m,
            float(np.corrcoef(t2, e2)[0, 1]),
            float(np.sqrt(np.mean((e2 - t2) ** 2))),
            float(np.mean(np.abs(e2 - t2))),
            float(step_truth),
            float(step_est),
        ))
    return rows


def report(name, truth, est, loaded):
    print(f"\n=== {name} vs MuJoCo ground truth (tau_ext, N·m) ===")
    print(f"{'joint':15} {'corr':>7} {'RMSE':>8} {'MAE':>8} {'gt step':>9} {'est step':>9}")
    for m, c, rse, mae, st, se in score(truth, est, loaded):
        print(f"{m:15} {c:7.3f} {rse:8.3f} {mae:8.3f} {st:+9.3f} {se:+9.3f}")


def main() -> int:
    parity, truth, loaded = collect()
    print(f"collected {len(parity)} samples @ {HZ} Hz; payload {PAYLOAD_KG*1000:.0f} g "
          f"for {loaded.sum()/HZ:.1f} s")
    rc = 0
    # 'ekf' and 'lpf' are the two methods this pipeline offers. NEXT is not a pipeline
    # method -- it is replayed offline from its own checkpoint (scripts/next_free_pipeline.py),
    # so it is reported as a follow-up rather than silently dropped.
    for method in ("ekf", "lpf"):
        try:
            est = run_estimator(parity, method)
        except Exception as exc:
            print(f"\n{method.upper()}: SKIPPED -- {type(exc).__name__}: {exc}")
            continue
        report(method.upper(), truth, est, loaded)
    print("\nNOTE: NEXT is not a pipeline method (valid: ekf, lpf). It is replayed offline "
          "from its own checkpoint; grading it against this same stream is the follow-up.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
