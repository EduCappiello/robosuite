"""
SOARM101 canonical physical specification — single source of truth.

All numeric values are derived from:
  - robosuite/models/assets/robots/SOARM101/SO101/soarm_with_sensor.xml  (arm)
  - robosuite/models/assets/grippers/so101_gripper.xml                   (gripper)

lerobot-private's CI uses validate_against_mjcf() to assert consistency.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINT_NAMES = JOINT_NAMES + [GRIPPER_JOINT]

# ---------------------------------------------------------------------------
# Joint limits [rad]  (from soarm_with_sensor.xml / so101_gripper.xml)
# ---------------------------------------------------------------------------

JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (-1.9198621771937616,  1.9198621771937634),
    "shoulder_lift": (-1.7453292519943224,  1.7453292519943366),
    "elbow_flex":    (-1.69,                1.69),
    "wrist_flex":    (-1.6580628494556928,  1.6580627293335335),
    "wrist_roll":    (-2.7438472969992493,  2.841206309382605),
    "gripper":       (-0.17453297762778586, 1.7453291995659765),
}

# ---------------------------------------------------------------------------
# STS3215 servo dynamics (class="sts3215" in XML)
# ---------------------------------------------------------------------------

JOINT_DAMPING      = 0.60    # N·m·s/rad  — dof_damping
JOINT_FRICTIONLOSS = 0.052   # N·m        — dof_frictionloss
JOINT_ARMATURE     = 0.028   # kg·m²      — dof_armature

SERVO_KP           = 998.22  # position actuator gain  [N·m/rad]
SERVO_KV           = 2.731   # velocity damping gain   [N·m·s/rad]
SERVO_FORCE_RANGE  = (-2.94, 2.94)   # N·m, from forcerange in XML

# ---------------------------------------------------------------------------
# Motor torque constant
# Callers (lerobot-private) should supply a per-robot-id calibrated kt.
# This default is a reasonable starting point for STS3215 at rated load.
# ---------------------------------------------------------------------------

KT_DEFAULT = 0.5   # N·m/A

# ---------------------------------------------------------------------------
# STS3215 servo register-conversion constants (parity-tier raw export)
#
# These convert a joint torque [N·m] back into the raw servo registers a real
# Feetech STS3215 would report (current_raw, load_raw), so a sim-exported
# parity-tier sample round-trips through lerobot-private's current->torque
# conversion.  They MUST stay numerically identical to
#   lerobot-private/.../so_follower/dynamics/motor_parameters.py
# otherwise the round-trip carries a multiplicative scale error.
#
# Forward (lerobot) path, per motor:
#   i_A         = current_raw * STS3215_CURRENT_RAW_TO_mA / 1000
#   tau_joint   = i_A * STS3215_TORQUE_CONSTANT_mNm_PER_A * (1/gear_ratio) / 1000
#   sign        = -sign(load_raw)   (fallback sign(velocity))
# servo_raw.py inverts exactly this.  SOARM101-specific (generalisation deferred).
# ---------------------------------------------------------------------------

STS3215_CURRENT_RAW_TO_mA         = 6.5         # raw current ADC units -> mA
STS3215_TORQUE_CONSTANT_mNm_PER_A = 31.8        # motor torque constant [mNm/A], pre-gearing
STS3215_GEAR_RATIO                = 1.0 / 345.0  # follower: 1/345 reduction on every joint
STS3215_STALL_CURRENT_A           = 3.0         # datasheet stall current; current clamp ceiling

RAD_TO_DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_mjcf(model) -> None:
    """
    Assert that the loaded MuJoCo model matches this canonical spec.

    Args:
        model: mujoco.MjModel already loaded from soarm_with_sensor.xml.

    Raises:
        AssertionError: if any value diverges beyond tolerance.
    """
    import mujoco

    TOL_ANGLE  = 1e-4  # rad
    TOL_INERTIA = 1e-6  # kg·m²

    for jname, (lo, hi) in JOINT_LIMITS.items():
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except Exception:
            # joint might be prefixed (e.g. robot0_shoulder_pan) when loaded via robosuite
            continue
        if jid < 0:
            continue
        actual_lo = float(model.jnt_range[jid, 0])
        actual_hi = float(model.jnt_range[jid, 1])
        assert abs(actual_lo - lo) < TOL_ANGLE, (
            f"Joint {jname} lower limit mismatch: canonical={lo:.6f} mjcf={actual_lo:.6f}"
        )
        assert abs(actual_hi - hi) < TOL_ANGLE, (
            f"Joint {jname} upper limit mismatch: canonical={hi:.6f} mjcf={actual_hi:.6f}"
        )

    for jname in ALL_JOINT_NAMES:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except Exception:
            continue
        if jid < 0:
            continue
        dof = int(model.jnt_dofadr[jid])
        actual_damping      = float(model.dof_damping[dof])
        actual_frictionloss = float(model.dof_frictionloss[dof])
        actual_armature     = float(model.dof_armature[dof])
        assert abs(actual_damping - JOINT_DAMPING) < 1e-4, (
            f"Joint {jname} damping mismatch: canonical={JOINT_DAMPING} mjcf={actual_damping}"
        )
        assert abs(actual_frictionloss - JOINT_FRICTIONLOSS) < 1e-5, (
            f"Joint {jname} frictionloss mismatch: canonical={JOINT_FRICTIONLOSS} mjcf={actual_frictionloss}"
        )
        assert abs(actual_armature - JOINT_ARMATURE) < TOL_INERTIA, (
            f"Joint {jname} armature mismatch: canonical={JOINT_ARMATURE} mjcf={actual_armature}"
        )
