"""
read_motor_signals(robot, kt) -> dict

Returns the raw motor signal schema consumed by lerobot-private.
Callable per physics substep (not only per control step).

Schema:
    pos            (6,) float64  — joint positions [rad], arm×5 then gripper×1
    vel_hw         (6,) float64  — joint velocities [rad/s]
    current        (6,) float64  — motor current [A], back-computed via kt
    current_signed (6,) float64  — current with sign tracking velocity direction
    load           (6,) float64  — normalised servo load in [-1, 1]
    dt             float         — this sample's physics timestep [s]
"""

from __future__ import annotations

import numpy as np

from robosuite_private.robot_spec.canonical import SERVO_FORCE_RANGE


def read_motor_signals(robot, kt: float) -> dict:
    """
    Read motor signals from a robosuite SingleArmRobot at the current sim state.

    Args:
        robot: a robosuite robot instance (e.g. env.robots[0]) that has been
               set up via setup_references().  Arm joint arm must be "right".
        kt:    motor torque constant [N·m/A].  Use KT_DEFAULT from canonical.py
               as a starting point; pass a calibrated value per robot_id in
               production.

    Returns:
        dict with keys: pos, vel_hw, current, current_signed, load, dt.
    """
    d = robot.sim.data

    # ---- index lists set up by robot.setup_references() -----------------
    arm_pos_idx  = robot._ref_joint_pos_indexes          # list[int], len=5
    arm_vel_idx  = robot._ref_joint_vel_indexes          # list[int], len=5 (nv)
    grip_pos_idx = robot._ref_gripper_joint_pos_indexes["right"]   # list[int], len=1
    grip_vel_idx = robot._ref_gripper_joint_vel_indexes["right"]   # list[int], len=1

    all_pos_idx = list(arm_pos_idx)  + list(grip_pos_idx)
    all_vel_idx = list(arm_vel_idx)  + list(grip_vel_idx)

    # ---- raw kinematics -------------------------------------------------
    pos    = np.array([d.qpos[i] for i in all_pos_idx], dtype=np.float64)
    vel_hw = np.array([d.qvel[i] for i in all_vel_idx], dtype=np.float64)

    # ---- generalized actuator force (= commanded torque for pos actuators) --
    # qfrc_actuator[nv] is the generalised force contributed by actuators.
    tau_act = np.array(d.qfrc_actuator)[all_vel_idx].astype(np.float64)

    # ---- derived signals -------------------------------------------------
    current        = tau_act / kt
    # sign tracks velocity direction so downstream can reconstruct polarity
    current_signed = current * np.sign(vel_hw + 1e-12)
    # normalised load: ratio of commanded torque to rated peak torque
    load = tau_act / SERVO_FORCE_RANGE[1]

    return {
        "pos":            pos,
        "vel_hw":         vel_hw,
        "current":        current,
        "current_signed": current_signed,
        "load":           load,
        "dt":             float(robot.sim.model.opt.timestep),
    }
