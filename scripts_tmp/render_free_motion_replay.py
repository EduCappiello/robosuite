#!/usr/bin/env python
"""Render the free-motion replay so the failure is visible, not inferred.

Two views side by side, same timestep:
  LEFT   what the SIM actually did (its own qpos after stepping under the recorded command)
  RIGHT  what the recording ASKED for (the recorded pose, set directly)
The gap between them is the tracking failure; the bar strip under each frame is the per-joint
actuator torque against the +/-2.942 N*m forcerange, red when pinned.

Measured cause this is meant to show (2026-11-20): the sim actuator is at the forcerange in
75 of 80 sampled steps, tracking gap up to 108 deg, and GT tau_ext_contact / tau_ext_jac are
exactly 0 while max|tau_ext| lands on 2.94 -- so tau_ext is the clipping deficit, not a load.

Run (needs the GPU free; see render_with_gpu_handoff.sh):
  MUJOCO_GL=egl conda run -n lerobot python scripts_tmp/render_free_motion_replay.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# MUJOCO_GL is set only when we actually render: forcing it makes even the CPU smoke test
# try to build a GL context and die in PyOpenGL.
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from next_free_pipeline import HISTORY, at_hz, load_stream  # noqa: E402

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower  # noqa: E402

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/next_free/20261118T060348Z_bus.npz")
    ap.add_argument("--hz", type=float, default=250.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--every", type=int, default=8, help="render every Nth step (250/8 ~= 31 fps)")
    ap.add_argument("--out", default="outputs/next_free/replay_render.mp4")
    ap.add_argument("--sim-hz", type=int, default=0,
                    help="sim control rate; 0 = same as --hz. The original driver default "
                         "was 30 Hz with kp=100/kv=20; it is now 500 Hz with kp=2400/kv=50.")
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--kv", type=float, default=None)
    ap.add_argument("--no-render", action="store_true",
                    help="run the replay and the stats without touching a GL context "
                         "(CPU-only smoke test; no GPU needed)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    s = load_stream(a.npz)
    s["t"] = np.asarray(s["t"], float) - float(np.asarray(s["t"], float)[0])
    s["keep"] = np.ones((len(s["t"]), 6))
    st = at_hz(s, a.hz)

    sim_hz = a.sim_hz or int(a.hz)
    kw = {}
    if a.kp is not None:
        kw["kp"] = a.kp
    if a.kv is not None:
        kw["kv"] = a.kv
    bot = RobosuiteSimFollower(RobosuiteSimConfig(
        id="render", control_freq=sim_hz, cameras=(), dynamics_robot_id="FT-02",
        emit_ground_truth=True, clear_task_objects=True, **kw))
    bot.connect()
    m, d = bot._model, bot._data

    # The log is read at --hz for the estimator-rate view, but the sim must be commanded at
    # ITS control rate, otherwise a 250 Hz command into a 30 Hz controller is not a replay at
    # 30 Hz, it is a 250 Hz command the controller never sees.
    sim_hz = a.sim_hz or int(a.hz)
    sc = at_hz(s, float(sim_hz))
    i0c = int(a.start * sim_hz)
    nc = int(a.seconds * sim_hz)
    i0 = int(a.start * a.hz)
    n = int(a.seconds * a.hz)
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{x}")])
            for x in MOTORS]
    lim = float(np.max(np.abs(m.actuator_ctrlrange[:5])))

    do_render = not a.no_render
    if do_render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    rend = mujoco.Renderer(m, a.height, a.width) if do_render else None
    cam = mujoco.MjvCamera()
    # The arm is mounted ON the table, so its bodies sit near z = 0.92 (measured:
    # robot0_upper_arm at [-0.31, -0.021, 0.917]). A lookat at z=0.25 put the whole arm out
    # of frame and the two views rendered bit-identical (diff exactly 0.0) while the replay
    # stats kept reporting a 108 deg gap -- a silently useless video. Verified: with this
    # camera a 60 deg shoulder_lift change moves 13,492 pixels.
    cam.lookat[:] = [0.0, 0.0, 1.00]
    cam.distance, cam.elevation, cam.azimuth = 0.90, -20, 60

    def shot():
        rend.update_scene(d, camera=cam)
        return cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)

    def annotate(img, txt, tau, sat):
        img = img.copy()
        cv2.rectangle(img, (0, a.height - 92), (a.width, a.height), (24, 24, 24), -1)
        cv2.putText(img, txt, (8, a.height - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        bw = (a.width - 20) // 6
        for j in range(6):
            v = float(np.clip(tau[j], -lim, lim)) if np.isfinite(tau[j]) else 0.0
            frac = abs(v) / lim
            x = 10 + j * bw
            h = int((frac) * 42)
            col = (0, 0, 255) if (sat and frac > 0.995) else (90, 190, 90)
            cv2.rectangle(img, (x, a.height - 14), (x + bw - 6, a.height - 14 - h), col, -1)
            cv2.putText(img, f"{SHORT[j]} {tau[j]:+.2f}" if np.isfinite(tau[j]) else SHORT[j],
                        (x, a.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1,
                        cv2.LINE_AA)
        return img

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    eff_every = max(1, int(round(sim_hz / a.hz * a.every)))
    w = (imageio_ffmpeg.write_frames(str(out), (a.width * 2, a.height), fps=sim_hz / eff_every,
                                    codec="libx264", pix_fmt_in="bgr24",
                                    output_params=["-crf", "22", "-preset", "veryfast"])
              if do_render else None)
    if do_render:
        w.send(None)

    def cmd_at(i):
        p = sc["pos"][i]
        g = {f"{x}.pos": float(p[j]) for j, x in enumerate(MOTORS)}
        g["gripper.pos"] = float(np.clip((p[5] + 10) / 1.1, 0.0, 100.0))
        return g

    for _ in range(HISTORY):
        bot.send_action(cmd_at(i0c))

    sat_steps = 0
    gaps = []
    for i in range(nc):
        bot.send_action(cmd_at(i0c + i))
        if i % max(1, int(round(sim_hz / a.hz * a.every))):
            continue
        left = shot() if do_render else None
        # Right view: the pose the recording asked for, set directly (no controller).
        # MUST be restored: without this the next send_action starts from the pose we just
        # teleported the arm to, and the replay silently stops being a replay.
        keep_q = d.qpos.copy(); keep_v = d.qvel.copy(); keep_a = d.qacc.copy()
        for j in range(6):
            d.qpos[qadr[j]] = np.deg2rad(sc["pos"][i0c + i, j])
        mujoco.mj_forward(m, d)
        right = shot() if do_render else None
        d.qpos[:] = keep_q; d.qvel[:] = keep_v; d.qacc[:] = keep_a
        mujoco.mj_forward(m, d)
        tau = np.asarray(d.ctrl[:6], dtype=np.float64)
        sat = bool(np.any(np.abs(tau[:5]) >= 0.995 * lim))
        sat_steps += int(sat)
        gap = float(np.max(np.abs(np.degrees([d.qpos[qadr[j]] for j in range(5)])
                                  - sc["pos"][i0c + i, :5])))
        gaps.append(gap)
        if not do_render:
            continue
        tag = f"t={i / sim_hz:5.2f}s  max gap {gap:6.1f} deg  |tau|>=limit: {'YES' if sat else 'no'}"
        if not do_render:
            continue
        frame = np.hstack([
            annotate(left, f"SIM actual   {tag}", tau, sat),
            annotate(right, f"RECORDED pose (the command)   forcerange +/-{lim:.2f} N*m", tau, sat),
        ])
        w.send(np.ascontiguousarray(frame))

    if do_render:
        w.close()
    bot.disconnect()
    print(f"sim control {sim_hz} Hz kp={bot.config.kp} kv={bot.config.kv}; "
          f"log read at {a.hz:.0f} Hz")
    print(f"wrote {out}  ({len(gaps)} frames)")
    print(f"saturated frames {sat_steps}/{len(gaps)}  max gap {max(gaps):.1f} deg  "
          f"mean gap {np.mean(gaps):.1f} deg")


if __name__ == "__main__":
    main()
