"""
Parity-tier exporter tests: to_servo_raw (SI motor signals -> raw STS3215 registers).

The decisive property (R6 / T14) is that a sim sample round-trips through
lerobot's *forward* STS conversion with no scale error.  We replicate that
forward path inline here from the SAME canonical constants — the test would
catch any drift between to_servo_raw's inverse and the constants the estimator
trusts, without importing lerobot (boundary rule).
"""

import numpy as np
import pytest

from robosuite_private.motor_signals import to_servo_raw
from robosuite_private.robot_spec import (
    SERVO_FORCE_RANGE,
    STS3215_CURRENT_RAW_TO_mA,
    STS3215_GEAR_RATIO,
    STS3215_STALL_CURRENT_A,
    STS3215_TORQUE_CONSTANT_mNm_PER_A,
)

_KEYS = {"pos_deg", "vel_deg_s", "current_raw", "load_raw"}


def _lerobot_forward_torque(current_raw, load_raw, dq=None):
    """Mirror of lerobot compute_motor_torque_signed (STS3215 forward path)."""
    i_A = current_raw * STS3215_CURRENT_RAW_TO_mA / 1000.0
    tau_mag = i_A * STS3215_TORQUE_CONSTANT_mNm_PER_A * (1.0 / STS3215_GEAR_RATIO) / 1000.0
    # Feetech load opposes motor torque: sign = -sign(load_raw); fall back to sign(dq).
    sign = -np.sign(load_raw)
    if dq is not None:
        sign = np.where(load_raw == 0.0, np.sign(dq), sign)
    return sign * tau_mag


def _make_signals(pos, vel, load):
    return {
        "pos": np.asarray(pos, dtype=np.float64),
        "vel_hw": np.asarray(vel, dtype=np.float64),
        "load": np.asarray(load, dtype=np.float64),
    }


# --------------------------------------------------------------------------- T13

def test_servo_raw_schema():
    sig = _make_signals(np.zeros(6), np.zeros(6), np.zeros(6))
    out = to_servo_raw(sig)
    assert set(out.keys()) == _KEYS
    for k, arr in out.items():
        assert isinstance(arr, np.ndarray), f"{k} must be ndarray"
        assert arr.shape == (6,), f"{k} shape {arr.shape} != (6,)"
        assert arr.dtype == np.float64, f"{k} dtype {arr.dtype} != float64"


def test_servo_raw_angle_units():
    """pos_deg / vel_deg_s are the SI radians inputs converted to degrees."""
    pos = np.array([0.0, np.pi / 2, -np.pi, 0.1, -0.1, 0.0])
    vel = np.array([1.0, -1.0, 0.0, 2.0, -2.0, 0.5])
    out = to_servo_raw(_make_signals(pos, vel, np.zeros(6)))
    np.testing.assert_allclose(out["pos_deg"], np.degrees(pos))
    np.testing.assert_allclose(out["vel_deg_s"], np.degrees(vel))


def test_servo_raw_current_unsigned_load_signed():
    """current_raw is unsigned (magnitude); polarity lives in load_raw."""
    load = np.array([0.3, -0.3, 0.5, -0.5, 0.0, 0.1])
    out = to_servo_raw(_make_signals(np.zeros(6), np.ones(6), load))
    assert np.all(out["current_raw"] >= 0.0), "current_raw must be unsigned"
    # load_raw = -load*1000, so its sign is the opposite of the joint-torque sign.
    np.testing.assert_allclose(out["load_raw"], -load * 1000.0)


# --------------------------------------------------------------------------- T14 (guards R6)

def test_servo_raw_torque_roundtrip():
    """to_servo_raw -> lerobot forward STS conversion recovers the joint torque (R6)."""
    # Stay well within the servo force range so the stall clamp never engages.
    load = np.array([0.25, -0.25, 0.4, -0.4, 0.05, -0.6])
    dq = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # nonzero velocities for sign fallback
    out = to_servo_raw(_make_signals(np.zeros(6), dq, load))

    tau_recovered = _lerobot_forward_torque(out["current_raw"], out["load_raw"], dq=dq)
    tau_joint = load * SERVO_FORCE_RANGE[1]  # the original joint torque [N·m]
    np.testing.assert_allclose(tau_recovered, tau_joint, atol=1e-9)


def test_servo_raw_zero_load_uses_velocity_sign():
    """At zero load the estimator falls back to dq sign; magnitude is zero either way."""
    out = to_servo_raw(_make_signals(np.zeros(6), np.ones(6), np.zeros(6)))
    np.testing.assert_allclose(out["current_raw"], np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(out["load_raw"], np.zeros(6), atol=1e-12)


def test_servo_raw_stall_clamp():
    """A torque beyond the drawable stall current is clamped, not extrapolated."""
    nm_per_A = (STS3215_TORQUE_CONSTANT_mNm_PER_A / 1000.0) * (1.0 / STS3215_GEAR_RATIO)
    max_tau = STS3215_STALL_CURRENT_A * nm_per_A           # torque at stall current
    huge_load = (2.0 * max_tau / SERVO_FORCE_RANGE[1]) * np.ones(6)  # 2x over stall
    out = to_servo_raw(_make_signals(np.zeros(6), np.ones(6), huge_load))
    max_current_raw = STS3215_STALL_CURRENT_A * 1000.0 / STS3215_CURRENT_RAW_TO_mA
    assert np.all(out["current_raw"] <= max_current_raw + 1e-6), "current_raw exceeds stall"
