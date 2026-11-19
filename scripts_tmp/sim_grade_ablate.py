#!/usr/bin/env python
"""Diagnose why the EKF's tau_ext over-responds to a payload in the sim.

Three separate things were tangled in the first grading run:

  (1) DEFINITION. ground_truth.ext.tau_ext is tau_ext_residual =
      tau_motor - G - C - M*qdd + tau_friction, computed from the SIM's own model. If the
      payload is added by mutating model.body_mass, the payload is INSIDE that model, so
      the gt tier reports ~nothing while the EKF (whose model has no payload) legitimately
      reports the payload's weight+inertia as tau_ext. That is a definitional offset, not
      estimator error.
      Fix: apply the payload as data.xfrc_applied on the jaw. Then the sim model stays
      unloaded, tau_ext_jac = -J^T w is exact truth, and both sides mean the same thing.

  (2) LOAD-DEPENDENT FRICTION. The sim grows dof_frictionloss = Fc + kappa*|G| under load.
      If the EKF's kappa differs, the mismatch lands in tau_ext. Ablate it.

  (3) ARMATURE. The sim has armature 0.028 on every joint; the EKF models 0. The unmodelled
      term is exactly armature * qdd, so measure it rather than argue about it.

RESULTS (2026-11-19, mid-range pose, 0.4 Hz / 8 deg sine, 0.2 kg for 3 s):

  (3) ARMATURE IS DEAD. arm*qdd steps by -0.0006..+0.0032 N*m. Two orders of magnitude
      below the effect being explained. Stop citing it.

  (2) LOAD-DEPENDENT FRICTION IS ALSO NOT THE CAUSE. dof_frictionloss steps by at most
      0.054 N*m (lift 0.298 -> 0.352; elbow 0.057 -> 0.041), and dPassive is essentially
      identical with load_dependent_friction ON (-0.458/-0.503/-0.451) and OFF
      (-0.454/-0.496/-0.447). The kappa*|G| term is real but far too small.

  WHAT IT ACTUALLY IS: viscous damping acting on a transient velocity shift. Under the
      applied wrench dq_mean steps by +0.206 rad/s (elbow) and the identified B turns that
      into ~0.5 N*m of passive torque -- matching dPassive (-0.45..-0.50) and the EKF error
      (+0.445..+0.485) in magnitude and sign. The payload's true external torque at this
      pose is only +0.027/-0.138/-0.201 N*m (tau_ext_jac).

  BENCHMARK WARNING: ground_truth.ext.*.tau_ext is tau_ext_residual when no applied wrench
      is found, and tau_ext_jac when one is. They differ by up to 1.49 N*m here. The
      residual is the quantity the estimator computes, so score the estimator against the
      RESIDUAL; jac is the physical applied-load torque and is the right target only for a
      settled, quasi-static window. Scoring the EKF against jac in a moving scenario
      measures the transient, not the estimator.

Run:
  cd ~/Documents/dev/robosuite
  PYTHONPATH=$PWD:~/Documents/dev/lerobot-1-tg/src \
    conda run -n lerobot --no-capture-output python -u scripts_tmp/sim_grade_ablate.py
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
SECONDS = 8.0
T0, T1 = 2.0, 5.0
PAYLOAD_KG = 0.200
G = 9.80665
ARMATURE = 0.028


def collect(variant: str):
    """variant: 'wrench' (xfrc_applied), 'mass' (body_mass), 'wrench_noloadfric'."""
    load_dep = variant != "wrench_noloadfric"
    bot = RobosuiteSimFollower(RobosuiteSimConfig(
        id="sim", control_freq=HZ, dynamics_robot_id="FT-02", emit_ground_truth=True,
        load_dependent_friction=load_dep))
    bot.connect()
    model, data = bot._model, bot._data
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_moving_jaw")
    base_mass = float(model.body_mass[jaw])
    # Per-joint dof addresses. body_dofadr[jaw] is the gripper slide, NOT the arm dofs --
    # reading through it silently samples the wrong joints.
    arm_dof = [int(model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{m}")]) for m in ARM]

    obs = bot.get_observation()
    init = {m: float(obs[f"{m}.pos"]) for m in MOTORS}
    # Command from a mid-range pose. The merged init pose sits at shoulder_lift -99.4 deg
    # inside a +-100 deg range, so a +-8 deg sine from there rides the lower limit and the
    # limit constraint torque (absent from the residual formula) swamps the payload signal.
    mid = dict(init)
    mid["shoulder_lift"] = -60.0
    mid["elbow_flex"] = -40.0
    for _ in range(500):
        bot.send_action({f"{m}.pos": mid[m] for m in MOTORS})
    n = int(HZ * SECONDS)
    parity, gt, loaded, qacc_rows, fric_rows, grav_rows, lim_rows = [], [], [], [], [], [], []
    for i in range(n):
        t = i / HZ
        on = T0 <= t < T1
        loaded.append(on)
        data.xfrc_applied[:] = 0.0
        model.body_mass[jaw] = base_mass
        if variant in ("wrench", "wrench_noloadfric") and on:
            # Gravity-equivalent wrench on the jaw, applied at the body origin in world
            # frame. The sim model stays unloaded, so the payload is external to it.
            data.xfrc_applied[jaw, 0] = 0.0
            data.xfrc_applied[jaw, 1] = 0.0
            data.xfrc_applied[jaw, 2] = -PAYLOAD_KG * G
        elif variant == "mass" and on:
            model.body_mass[jaw] = base_mass + PAYLOAD_KG
        mujoco.mj_forward(model, data)

        obs = bot.get_observation()
        parity.append({k: v for k, v in obs.items() if not k.startswith("ground_truth")})
        gt.append({
            "jac": {m: float(obs[f"ground_truth.ext.{m}.tau_ext_jac"]) for m in MOTORS},
            "res": {m: float(obs[f"ground_truth.ext.{m}.tau_ext_residual"]) for m in MOTORS},
        })
        qacc_rows.append([float(data.qacc[d]) for d in arm_dof])
        fric_rows.append([float(data.qfrc_passive[d]) for d in arm_dof])
        grav_rows.append([float(-data.qfrc_bias[d]) for d in arm_dof])
        # Joint-limit constraint torque: NOT in the residual formula, so it silently
        # contaminates both the gt residual and the EKF's tau_ext when the pose rides a limit.
        lim_rows.append([float(data.qfrc_constraint[d]) for d in arm_dof])

        s = 8.0 * np.sin(2 * np.pi * 0.4 * t)
        goal = {f"{m}.pos": mid[m] + (s if m in ("shoulder_lift", "elbow_flex") else 0.0)
                for m in MOTORS}
        bot.send_action(goal)
    bot.disconnect()
    return (parity, gt, np.array(loaded), np.array(qacc_rows), np.array(fric_rows),
            np.array(grav_rows), np.array(lim_rows))


def run_estimator(parity, method="ekf"):
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


def step_of(series, loaded):
    """Mean during the payload window minus mean outside it."""
    a = series[loaded]
    b = series[~loaded]
    return float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")


def report(variant, gt, est, loaded, qacc, fric, grav, lim):
    print(f"\n===== {variant} =====")
    print("self-consistency  |tau_ext_residual - tau_ext_jac| max, loaded window:")
    l = loaded
    for m in ARM:
        r = np.array([g["res"][m] for g in gt])
        j = np.array([g["jac"][m] for g in gt])
        print(f"  {m:15} {np.max(np.abs(r[l] - j[l])):.4f} N*m")
    print(f"\n{'joint':15} {'gt jac':>9} {'EKF':>9} {'err':>9} | "
          f"{"dG(sim)":>9} {'dPassive':>10} {'arm*Qdd':>9} {'dLimit':>9}")
    for k, m in enumerate(ARM):
        j = np.array([g["jac"][m] for g in gt])
        e = np.array([r[m] for r in est])
        dG = step_of(grav[:, k], loaded)
        dF = step_of(fric[:, k], loaded)
        # unmodelled inertial term the EKF cannot see: armature * qdd
        arm = step_of(ARMATURE * qacc[:, k], loaded)
        dL = step_of(lim[:, k], loaded)
        print(f"{m:15} {step_of(j, loaded):+9.3f} {step_of(e, loaded):+9.3f} "
              f"{step_of(e, loaded) - step_of(j, loaded):+9.3f} | "
              f"{dG:+9.3f} {dF:+10.3f} {arm:+9.4f} {dL:+9.3f}")
    print("  dG/dFric are the SIM's own gravity and friction terms; under xfrc_applied they")
    print("  should NOT change (the payload is not in the model). arm*Qdd is the term the")
    print("  EKF cannot model because it uses armature 0.")


def main() -> None:
    for variant in ("wrench", "wrench_noloadfric", "mass"):
        parity, gt, loaded, qacc, fric, grav, lim = collect(variant)
        est = run_estimator(parity, "ekf")
        report(variant, gt, est, loaded, qacc, fric, grav, lim)


if __name__ == "__main__":
    main()
