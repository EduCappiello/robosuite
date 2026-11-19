#!/usr/bin/env python
"""Grade NEXT against MuJoCo tau_ext ground truth, A/B against the EKF on one stream.

NEXT is not a pipeline method (the pipeline offers ekf/lpf), so it is stepped here through
the same NextEstimator the live bus uses, on the same parity-tier samples.

The contract that bit us before: NextEstimator.tau_ext is already tau_measured - tau_free
(Eq. 2), i.e. an EXTERNAL torque. tau_free is the predicted FREE-space motor torque. Do not
subtract tau_ext again, and do not treat tau_free as the answer.

RATE CAVEAT: the vendored checkpoints were trained at 100 and 250 Hz. The sim stream is
500 Hz, so NEXT's H-window covers half the time it saw in training. Both are measured here:
native 500 Hz and decimated 250 Hz, so the rate mismatch is a measured number rather than
an assumption.

Run:
  cd ~/Documents/dev/robosuite
  PYTHONPATH=$PWD:~/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_grade_next.py
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower
from lerobot.robots.so_follower.dynamics.motor_parameters import SO101_FOLLOWER_GEAR_RATIOS
from lerobot.robots.so_follower.dynamics.next.live_estimator import NextEstimator
from lerobot.robots.so_follower.dynamics.torque_estimator_processor import (
    make_so_torque_pipeline_steps,
)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = MOTORS[:5]
HZ = 500
SECONDS = 12.0
T_ON, RAMP_S, T_OFF = 3.0, 1.5, 9.0
SETTLE_S = 2.0
PAYLOAD_KG = 0.200
G = 9.80665
LEROBOT = os.path.expanduser("~/Documents/dev/lerobot-1-tg")
CKPTS = {"next@250Hz-native": "next_250hz", "next@500Hz-rate-mismatch": "next_250hz"}


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
    rows = {k: [] for k in ("pos", "vel", "cur", "load", "goal", "res", "jac")}
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
        rows["pos"].append([float(obs[f"{m}.pos"]) for m in MOTORS])
        rows["vel"].append([float(obs[f"{m}.vel_hw"]) for m in MOTORS])
        rows["cur"].append([float(obs[f"{m}.current_raw"]) for m in MOTORS])
        rows["load"].append([float(obs[f"{m}.load_raw"]) for m in MOTORS])
        rows["goal"].append([mid[m] for m in MOTORS])
        rows["res"].append([float(obs[f"ground_truth.ext.{m}.tau_ext_residual"]) for m in MOTORS])
        rows["jac"].append([float(obs[f"ground_truth.ext.{m}.tau_ext_jac"]) for m in MOTORS])
        bot.send_action({f"{m}.pos": mid[m] for m in MOTORS})
    bot.disconnect()
    return {k: np.array(v) for k, v in rows.items()}


def run_next(rows, ckpt_name, decimate=1):
    run_dir = os.path.join(LEROBOT, "outputs/next_free/pipeline", ckpt_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    est = NextEstimator(run_dir, MOTORS, SO101_FOLLOWER_GEAR_RATIOS, device="cpu")
    step = max(1, int(decimate))
    out = []
    for i in range(0, len(rows["pos"]), step):
        e = est.step(
            np.deg2rad(rows["pos"][i]), np.deg2rad(rows["vel"][i]),
            rows["cur"][i], rows["load"][i],
            goal_rad=np.deg2rad(rows["goal"][i]),
        )
        out.append(np.asarray(e.tau_ext, dtype=float))
    return np.array(out)


def run_ekf(rows, decimate=1):
    from lerobot.scripts.lerobot_teleoperate import make_default_processors

    pipeline = make_default_processors()[2]
    pipeline.steps = list(pipeline.steps) + make_so_torque_pipeline_steps(
        motor_names=MOTORS, robot_type="so101_follower", robot_id="FT-02",
        estimation_method="ekf")
    step = max(1, int(decimate))
    keys = ("pos", "vel", "cur", "load")
    out = []
    for i in range(0, len(rows["pos"]), step):
        obs = {}
        for j, m in enumerate(MOTORS):
            obs[f"{m}.pos"] = rows["pos"][i, j]
            obs[f"{m}.vel_hw"] = rows["vel"][i, j]
            obs[f"{m}.current_raw"] = rows["cur"][i, j]
            obs[f"{m}.load_raw"] = rows["load"][i, j]
        o = pipeline(obs)
        out.append([float(o[f"{m}.estimated/tau_ext"]) for m in MOTORS])
    return np.array(out)


def windows(decimate=1):
    step = max(1, int(decimate))
    t = np.arange(0, int(HZ * SECONDS), step) / HZ
    base = (t >= 0.5) & (t < T_ON)
    load = (t >= T_ON + RAMP_S + SETTLE_S) & (t < T_OFF)
    return base, load


def score(name, est, rows, decimate=1):
    base, load = windows(decimate)
    res = rows["res"][::max(1, int(decimate))]
    jac = rows["jac"][::max(1, int(decimate))]
    print(f"\n=== {name} ===")
    print(f"{'joint':15} {'gt res step':>12} {'est step':>10} {'err':>8} | "
          f"{'gt jac step':>11} | {'corr':>7} {'RMSE':>7}")
    tot = []
    for k, m in enumerate(ARM):
        e = est[:, k]
        ok = np.isfinite(e)
        b, l = base & ok, load & ok
        if not (b.any() and l.any()):
            print(f"{m:15} {'warm-up / non-finite':>12}")
            continue
        st_r = res[l, k].mean() - res[b, k].mean()
        st_e = e[l].mean() - e[b].mean()
        st_j = jac[l, k].mean() - jac[b, k].mean()
        msk = b | l
        corr = float(np.corrcoef(res[msk, k], e[msk])[0, 1])
        rmse = float(np.sqrt(np.mean((e[msk] - res[msk, k]) ** 2)))
        print(f"{m:15} {st_r:+12.3f} {st_e:+10.3f} {st_e-st_r:+8.3f} | "
              f"{st_j:+11.3f} | {corr:7.3f} {rmse:7.3f}")
        tot.append(abs(st_e - st_r))
    if tot:
        print(f"mean |step error| over 5 arm joints: {np.mean(tot):.3f} N*m")


def main() -> None:
    rows = collect()
    print(f"stream: {len(rows['pos'])} samples @ {HZ} Hz; ramped 0.2 kg payload "
          f"{T_ON}..{T_OFF}s (ramp {RAMP_S}s, {SETTLE_S}s settle discarded)")

    score("EKF (reference, 500 Hz)", run_ekf(rows), rows)
    for label, ckpt in CKPTS.items():
        dec = 2 if label.endswith("native") else 1
        try:
            est = run_next(rows, ckpt, decimate=dec)
        except Exception as exc:
            print(f"\n=== {label} === SKIPPED: {type(exc).__name__}: {exc}")
            continue
        eff = HZ / max(1, dec)
        score(f"NEXT {label} (effective {eff:.0f} Hz, ckpt {ckpt})", est, rows, decimate=dec)


if __name__ == "__main__":
    main()
