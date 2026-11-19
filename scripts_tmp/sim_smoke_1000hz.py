#!/usr/bin/env python
"""SOARM101 sim smoke test: 1000 Hz physics / 500 Hz control + payload equivalence.

Run from the robosuite repo root with BOTH paths on PYTHONPATH:

  cd /home/ecappiell/Documents/dev/robosuite
  PYTHONPATH=$PWD:/home/ecappiell/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_smoke_1000hz.py

Simulation only. Never touches a serial port.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from lerobot.robots.robosuite_sim import RobosuiteSimConfig, RobosuiteSimFollower

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
G = 9.80665

RESULTS: list[tuple[str, str, str]] = []


def record(test: str, ok: bool, detail: str) -> None:
    RESULTS.append((test, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {test}: {detail}", flush=True)


def raw_model(m):
    """robosuite wraps mujoco.MjModel (binding_utils.MjModel); the mujoco C functions only
    accept the underlying dm-control object, which lives in ._model."""
    while not isinstance(m, mujoco.MjModel):
        m = m._model
    return m


def raw_data(d):
    """Same unwrap for MjData (._data)."""
    while not isinstance(d, mujoco.MjData):
        d = d._data
    return d


def dof_of(model, name: str) -> int:
    j = mujoco.mj_name2id(raw_model(model), mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{name}")
    assert j >= 0, f"joint robot0_{name} not found"
    return int(model.jnt_dofadr[j])


def joint_of(model, name: str) -> int:
    j = mujoco.mj_name2id(raw_model(model), mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{name}")
    assert j >= 0, f"joint robot0_{name} not found"
    return j


def hold(bot, n_steps: int, targets: dict[str, float] | None = None,
         motion: dict[str, float] | None = None):
    """Step n control steps. `targets` = absolute deg setpoints; `motion` = {joint: amplitude_deg}
    sine sweep at 0.5 Hz added on top of the target. Returns max|qacc| and NaN count."""
    qacc_max, qvel_max, nans = 0.0, 0.0, 0
    for i in range(n_steps):
        act = {}
        for m in MOTORS:
            base = targets[m] if targets else 0.0
            if motion and m in motion:
                base = base + motion[m] * np.sin(2 * np.pi * 0.5 * i / 500.0)
            act[f"{m}.pos"] = float(base)
        bot.send_action(act)
        d = bot.sim.env.sim.data
        for arr in (d.qacc, d.qvel, d.qfrc_bias):
            if not np.isfinite(arr).all():
                nans += 1
                break
        else:
            qacc_max = max(qacc_max, float(np.max(np.abs(d.qacc))))
            qvel_max = max(qvel_max, float(np.max(np.abs(d.qvel))))
    return qacc_max, qvel_max, nans


def main() -> int:
    print("=" * 78)
    print("SOARM101 sim smoke — 1000 Hz physics / 500 Hz control")
    print("=" * 78)

    # ---------------------------------------------------------------- T1/T2 identified
    cfg = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="FT-02",
                             emit_ground_truth=True, has_renderer=False)
    bot = RobosuiteSimFollower(cfg)
    bot.connect()
    env = bot.sim.env
    model = env.sim.model

    mt, ct = float(env.model_timestep), float(env.control_timestep)
    record("T1 rates", abs(mt - 0.001) < 1e-12 and env.control_freq == 500 and ct / mt == 2.0,
           f"physics={1/mt:.0f}Hz control={env.control_freq}Hz substeps={ct/mt:.0f}")

    print("\n-- T2 identified friction (dynamics_robot_id='FT-02') --")
    ident = {}
    for m in MOTORS:
        d = dof_of(model, m)
        ident[m] = (float(model.dof_armature[d]), float(model.dof_frictionloss[d]),
                    float(model.dof_damping[d]))
        print(f"   {m:14} armature={ident[m][0]:.4f} frictionloss={ident[m][1]:.4f} damping={ident[m][2]:.4f}")
    changed = abs(ident["shoulder_lift"][2] - 0.60) > 1e-3
    # Upstream joints_properties.xml sets armature 0.028 on the whole sts3215 class (all six
    # joints are the same servo). The sim keeps it; only the real-robot estimator uses 0.
    arm_ok = all(abs(ident[j][0] - 0.028) < 1e-6 for j in MOTORS)
    record("T2 identified friction + armature policy", changed and arm_ok,
           f"lift damping {ident['shoulder_lift'][2]:.4f} != XML 0.60; "
           f"armature 0.028 on all six joints={arm_ok}")

    # ---------------------------------------------------------------- T4 hold stability
    obs = bot.get_observation()
    init_deg = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    print(f"\n-- init pose (deg) -- {init_deg}")
    qacc_max, qvel_max, nans = hold(bot, 2000, init_deg)
    record("T4 hold 2000 steps @500Hz", nans == 0 and qacc_max < 2000,
           f"max|qacc|={qacc_max:.1f} max|qvel|={qvel_max:.2f} nan_steps={nans}")

    # ---------------------------------------------------------------- T5 sine motion
    # Command from MID-RANGE, not from the reset pose: the merged init pose puts
    # shoulder_lift at -99.4 deg inside a +-100 deg range, i.e. 0.6 deg from its lower
    # joint limit, so a +-10 deg sine from there measures limit contact rather than
    # tracking. Mid-range targets make T5 a genuine tracking test.
    mid_deg = dict(init_deg)
    mid_deg["shoulder_lift"] = -60.0
    mid_deg["elbow_flex"] = -40.0
    hold(bot, 500, mid_deg)          # settle at the mid-range pose first
    qacc_max, qvel_max, nans = hold(bot, 2000, mid_deg,
                                    motion={"shoulder_lift": 10.0, "elbow_flex": 10.0})
    q = env.sim.data.qpos
    exc = {}
    for m in ["shoulder_lift", "elbow_flex"]:
        jadr = int(model.jnt_qposadr[joint_of(model, m)])
        exc[m] = float(np.rad2deg(np.max(np.abs(q[jadr] - mid_deg[m] * np.pi / 180))))
    record("T5 sine motion 2000 steps", nans == 0 and qacc_max < 2000 and exc["shoulder_lift"] > 1.0,
           f"max|qacc|={qacc_max:.1f} nan={nans} excursion_deg={exc}")

    # ---------------------------------------------------------------- T6 realtime
    t0 = time.perf_counter()
    hold(bot, 2000, init_deg)
    dt_id = time.perf_counter() - t0
    bot.disconnect()

    cfg2 = RobosuiteSimConfig(id="sim", control_freq=500, dynamics_robot_id="NO_SUCH_ROBOT",
                              emit_ground_truth=True)
    bot2 = RobosuiteSimFollower(cfg2)
    bot2.connect()
    d = dof_of(bot2.sim.env.sim.model, "shoulder_lift")
    fb = (float(bot2.sim.env.sim.model.dof_armature[d]),
          float(bot2.sim.env.sim.model.dof_frictionloss[d]),
          float(bot2.sim.env.sim.model.dof_damping[d]))
    print(f"\n-- T3 fallback (no calib file) lift armature/frictionloss/damping = {fb}")
    # Armature policy is independent of the friction calibration: the arm joints are zeroed
    # whether or not a calibration file exists. Only frictionloss/damping fall back to XML.
    record("T3 fallback to XML defaults", abs(fb[0] - 0.028) < 1e-6 and abs(fb[1] - 0.052) < 1e-6
           and abs(fb[2] - 0.60) < 1e-6, f"lift armature={fb[0]} frictionloss={fb[1]} damping={fb[2]}")
    obs2 = bot2.get_observation()
    init2 = {m: float(obs2[f"{m}.pos"]) for m in MOTORS}
    t0 = time.perf_counter()
    hold(bot2, 2000, init2)
    dt_fb = time.perf_counter() - t0
    record("T6 realtime 2000 steps @500Hz", True,
           f"identified {2000/500/dt_id:.2f}x realtime ({dt_id:.2f}s), "
           f"fallback {2000/500/dt_fb:.2f}x ({dt_fb:.2f}s)")

    # ---------------------------------------------------------------- T7 payload
    # Add mass to the moving jaw (the TCP body) and read the simulated wrist FT sensor.
    # Real-robot equivalent: the 0/50/100/200 g box replays.
    print("\n-- T7 payload equivalence (mass on moving jaw -> wrist FT + joint gravity torque) --")
    m2 = raw_model(bot2.sim.env.sim.model)
    d2 = raw_data(bot2.sim.env.sim.data)
    jaw = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")
    assert jaw >= 0, "robot0_moving_jaw body not found"
    # T7 diagnostic: is the jaw inside the wrist FT site's subtree?  A MuJoCo <force site=...>
    # sensor reads the force transmitted through the site, i.e. only loads carried by the
    # subtree rooted at the body the site is attached to.
    ft_site = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SITE, "robot0_wrist_ft_site")
    site_body = int(m2.site_bodyid[ft_site]) if ft_site >= 0 else -1
    print(f"   wrist_ft_site id={ft_site} attached to body "
          f"'{mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, site_body) if site_body >= 0 else None}'")
    chain, b = [], jaw
    while b > 0:
        chain.append(mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, b))
        b = int(m2.body_parentid[b])
    chain.append(mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, 0))
    print(f"   jaw->world body chain: {' -> '.join(chain)}")
    base_mass = float(bot2.sim.env.sim.model.body_mass[jaw])
    sub_mass = sum(float(m2.body_mass[b]) for b in range(m2.nbody)
                   if int(m2.body_weldid[b]) == site_body)
    print(f"   moving_jaw base mass = {base_mass:.4f} kg; rigid mass at/below FT site = "
          f"{sub_mass:.4f} kg (weight {sub_mass * G:.4f} N)")

    rows = []
    for grams in (0, 50, 100, 200):
        kg = grams / 1000.0
        bot2.sim.env.sim.model.body_mass[jaw] = base_mass + kg
        mujoco.mj_forward(m2, d2)
        hold(bot2, 250, init2)  # settle
        o = bot2.get_observation()
        # tcp_wrench_ft is the raw MuJoCo <force site=wrist_ft_site> reading, expressed in the
        # SITE frame (quat "1 0 0 0" in the wrist_roll body frame), not the world frame.  The
        # payload weight is a world-frame vertical force, so the comparison must be done on a
        # frame-independent quantity: rotate to world with site_xmat, then use |F| (and Fz_world).
        f_site = np.array([o["ground_truth.ext.tcp_wrench_ft." + a]
                           for a in ("fx", "fy", "fz")], dtype=np.float64)
        R_site = np.array(d2.site_xmat, dtype=np.float64).reshape(m2.nsite, 3, 3)[ft_site]  # site -> world
        f_world = R_site @ f_site
        fz = float(o["ground_truth.ext.tcp_wrench_ft.fz"])
        # lever arm: horizontal distance from each joint axis to the jaw CoM
        xjaw = bot2.sim.env.sim.data.xpos[jaw]
        lev = {}
        for jm in ("shoulder_lift", "elbow_flex"):
            jid = joint_of(bot2.sim.env.sim.model, jm)
            axis = bot2.sim.env.sim.model.jnt_axis[jid]
            xj = bot2.sim.env.sim.data.xpos[
                int(bot2.sim.env.sim.model.jnt_bodyid[jid])]
            r = xjaw - xj
            lev[jm] = float(np.linalg.norm(r - np.dot(r, axis) * axis))
        rows.append((grams, fz, float(np.linalg.norm(f_world)), float(f_world[2]), lev, o))
        print(f"   {grams:3d} g  FT fz(site)={fz:+.3f} N  |F|={np.linalg.norm(f_world):.3f} N "
              f"Fz(world)={f_world[2]:+.3f} N   lever_lift={lev['shoulder_lift']:.4f} m "
              f"lever_elbow={lev['elbow_flex']:.4f} m")

    print(f"   site-z vs world-z cos = {float(R_site[2, 2]):+.4f} "
          f"(site-frame fz therefore sees only cos of the payload weight)")

    def fit(getter):
        base = getter(rows[0])
        d = np.array([getter(r) - base for r in rows[1:]])
        m = np.array([r[0] / 1000.0 for r in rows[1:]])
        sl = float((d @ m) / (m @ m))                          # through origin
        pred = m * G
        r2 = float(1 - np.sum((d - pred) ** 2) / np.sum((d - d.mean()) ** 2))
        rel = float(np.max(np.abs(d - pred) / pred))
        return d, sl, r2, rel

    d_site, sl_site, _, _ = fit(lambda r: r[1])
    d_mag, sl_mag, r2_mag, rel_mag = fit(lambda r: r[2])
    d_wz, sl_wz, r2_wz, rel_wz = fit(lambda r: r[3])
    print(f"   d|F| = {np.array2string(d_mag, precision=4)} N vs m*g = "
          f"{np.array2string(np.array([r[0] / 1000.0 for r in rows[1:]]) * G, precision=4)} N")
    # Pass criterion: the frame/sign-independent magnitude |F| must track m*g.  The signed
    # world-z slope is reported for the sign convention (MuJoCo force sensors report the
    # force the parent transmits INTO the subtree, i.e. it points UP, opposing gravity).
    record("T7 FT sensor reads payload weight",
           r2_mag > 0.98 and rel_mag < 0.15 and abs(sl_wz) > 0.85 * G,
           f"empty |F|={rows[0][2]:.3f} N; d|F|/dm={sl_mag:+.3f} N/kg (R^2={r2_mag:.4f}, "
           f"max rel err={rel_mag:.3f}); dFz_world/dm={sl_wz:+.3f} (R^2={r2_wz:.4f}, "
           f"rel err={rel_wz:.3f}); site-frame fz slope={sl_site:+.3f} "
           f"(R^2-limited: {fit(lambda r: r[1])[2]:.4f})")

    print("\n" + "=" * 78)
    print(f"{'TEST':38} {'RESULT':7} DETAIL")
    print("-" * 78)
    for t, s, det in RESULTS:
        print(f"{t:38} {s:7} {det}")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("-" * 78)
    print("VERDICT:", "ALL TESTS PASSED" if n_fail == 0 else f"{n_fail} TEST(S) FAILED")
    bot2.disconnect()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
