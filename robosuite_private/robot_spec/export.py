"""
export_robot_spec() — return the canonical SOARM101 spec as a plain dict.

Used by lerobot-private CI to assert consistency between this MJCF and
the Pinocchio URDF (or any other model representation).
"""

from __future__ import annotations

from robosuite_private.robot_spec.canonical import (
    ALL_JOINT_NAMES,
    GRIPPER_JOINT,
    JOINT_ARMATURE,
    JOINT_DAMPING,
    JOINT_FRICTIONLOSS,
    JOINT_LIMITS,
    JOINT_NAMES,
    KT_DEFAULT,
    SERVO_FORCE_RANGE,
    SERVO_KP,
    SERVO_KV,
    STS3215_CURRENT_RAW_TO_mA,
    STS3215_GEAR_RATIO,
    STS3215_STALL_CURRENT_A,
    STS3215_TORQUE_CONSTANT_mNm_PER_A,
)


def export_robot_spec() -> dict:
    """
    Return the canonical SOARM101 physical specification.

    Schema is stable — version it (add a 'version' key) if values change.
    """
    return {
        "version":          2,
        "robot":            "SOARM101",
        "joint_names":      JOINT_NAMES,
        "gripper_joint":    GRIPPER_JOINT,
        "all_joint_names":  ALL_JOINT_NAMES,
        "joint_limits":     {k: list(v) for k, v in JOINT_LIMITS.items()},
        "joint_damping":    JOINT_DAMPING,
        "joint_frictionloss": JOINT_FRICTIONLOSS,
        "joint_armature":   JOINT_ARMATURE,
        "servo_kp":         SERVO_KP,
        "servo_kv":         SERVO_KV,
        "servo_force_range": list(SERVO_FORCE_RANGE),
        "kt_default":       KT_DEFAULT,
        # STS3215 register-conversion constants (parity-tier raw export); these
        # must match lerobot-private's motor_parameters.py exactly.
        "sts3215_current_raw_to_mA":         STS3215_CURRENT_RAW_TO_mA,
        "sts3215_torque_constant_mNm_per_A": STS3215_TORQUE_CONSTANT_mNm_PER_A,
        "sts3215_gear_ratio":                STS3215_GEAR_RATIO,
        "sts3215_stall_current_A":           STS3215_STALL_CURRENT_A,
    }
