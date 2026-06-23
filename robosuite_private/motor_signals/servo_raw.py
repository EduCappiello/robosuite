"""
to_servo_raw(signals, *, spec=canonical) -> dict

Convert SI motor signals (from read_motor_signals) into the raw servo registers
a real Feetech STS3215 would report.  This is the parity tier: the signals that
exist on BOTH the real arm and the sim, and the exact inputs lerobot-private's
estimators consume.

Pure & numeric — no lerobot import, no key-naming.  The lerobot adapter is
responsible for naming the outputs `{motor}.pos`, `{motor}.current_raw`, etc.

Output schema (each (6,) float64, DOF order = 5 arm joints then gripper):
    pos_deg      — joint position [deg]   (real arm reports degrees when use_degrees=True)
    vel_deg_s    — joint velocity [deg/s]
    current_raw  — UNSIGNED raw current register (ADC units); sign lives in load_raw
    load_raw     — sign-magnitude load register; sign = −sign(motor torque)

Exact inverse of lerobot's compute_motor_torque_signed (motor_parameters.py):
    forward:  i_A = current_raw·k_mA/1000 ;  τ = i_A·k_τ·(1/gear)/1000 ;  sign = −sign(load_raw)
    here:     τ recovered from load·forcerange ;  i_A = τ/(k_τ/1000·1/gear) ;  current_raw = |i_A|·1000/k_mA
so a sim sample round-trips through the estimator with no scale error (risk R6).
"""

from __future__ import annotations

import numpy as np

from robosuite_private.robot_spec import canonical as _canonical


def to_servo_raw(signals: dict, *, spec=_canonical) -> dict:
    """
    Map an SI read_motor_signals() dict to raw STS3215 registers.

    Args:
        signals: dict with keys "pos" (rad), "vel_hw" (rad/s), "load"
                 (= joint torque / SERVO_FORCE_RANGE[1], dimensionless).
        spec: module/object exposing the STS3215_* constants and
              SERVO_FORCE_RANGE (defaults to robot_spec.canonical).

    Returns:
        dict with keys pos_deg, vel_deg_s, current_raw, load_raw.
    """
    pos = np.asarray(signals["pos"], dtype=np.float64)
    vel = np.asarray(signals["vel_hw"], dtype=np.float64)
    load = np.asarray(signals["load"], dtype=np.float64)

    # Recover joint torque [N·m]: load is normalized by the peak servo torque.
    tau_joint = load * spec.SERVO_FORCE_RANGE[1]

    # STS3215 inverse — joint torque -> motor current [A].
    # Use the STS torque/current constants, NOT the SI reader's generic kt, or
    # the round-trip carries a ~10^4 scale error (risk R6).
    nm_per_A = (spec.STS3215_TORQUE_CONSTANT_mNm_PER_A / 1000.0) * (1.0 / spec.STS3215_GEAR_RATIO)
    i_A = tau_joint / nm_per_A
    # Clamp to the physically drawable current (datasheet stall); inert within
    # the servo force range, so it never perturbs the round-trip.
    i_A = np.clip(i_A, -spec.STS3215_STALL_CURRENT_A, spec.STS3215_STALL_CURRENT_A)

    # Unsigned raw current register; polarity is carried separately by load_raw.
    current_raw = np.abs(i_A) * 1000.0 / spec.STS3215_CURRENT_RAW_TO_mA

    # Sign-magnitude load register.  Feetech load opposes motor torque, so the
    # estimator recovers motor sign as −sign(load_raw); with load ∝ +torque that
    # means load_raw = −load·1000 (a zero leaves the estimator to fall back to dq).
    load_raw = -load * 1000.0

    return {
        "pos_deg": np.degrees(pos),
        "vel_deg_s": np.degrees(vel),
        "current_raw": current_raw.astype(np.float64),
        "load_raw": load_raw.astype(np.float64),
    }
