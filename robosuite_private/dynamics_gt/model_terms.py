"""
compute_model_terms(model, data, robot) -> dict   ("ground_truth.model.*")

Rigid-body torque decomposition for the SOARM101, read live from a MuJoCo
simulation.  Every term is the exact analytic quantity the lerobot-private
estimator tries to reconstruct from motor signals, so residuals reflect
algorithm quality, not parameter mismatch.

Shared equation of motion (same convention as the estimator):

    M(q)·q̈ = τ_motor − (G(q) + C(q,q̇)·q̇) − friction − τ_ext

Schema (all (6,) float64, DOF order = 5 arm joints then gripper):
    tau_motor    — commanded actuator torque (qfrc_actuator)
    tau_gravity  — gravity torque G(q) at current q, q̇=0
    tau_coriolis — Coriolis + centrifugal  C(q,q̇)·q̇
    tau_inertial — inertial term  M(q)·q̈
    tau_friction — reconstructed Coulomb + viscous friction (APPLIED sign, negative)
    tau_model    — tau_gravity + tau_coriolis + tau_inertial + tau_friction
                   (matches lerobot's TorqueEstimate.tau_model)
"""

from __future__ import annotations

import numpy as np


def _arm_dof_indices(robot) -> list[int]:
    """Return nv-space DOF indices for all 6 joints (arm then gripper).

    Robot-agnostic: uses only the robot's own reference-index attributes, so the
    same function works for any robosuite SingleArm robot whose arm is "right".
    """
    return (
        list(robot._ref_joint_vel_indexes)
        + list(robot._ref_gripper_joint_vel_indexes["right"])
    )


def compute_model_terms(model, data, robot) -> dict:
    """
    Compute the rigid-body torque decomposition (ground_truth.model.*).

    Args:
        model: mujoco.MjModel (env.sim.model._model)
        data:  mujoco.MjData  (env.sim.data._data)
        robot: robosuite robot instance (env.robots[0])

    Returns:
        dict with keys tau_motor, tau_gravity, tau_coriolis, tau_inertial,
        tau_friction, tau_model — each (6,) float64.

    Notes:
        - tau_gravity is computed by zeroing q̇ on a scratch MjData and reading
          qfrc_bias (MuJoCo has no standalone G(q) call).  mj_forward is required
          because fwdVelocity depends on fwdPosition having run.
        - tau_friction is reconstructed from model.dof_frictionloss / dof_damping
          using the SAME functional form as the estimator.  Do NOT read
          qfrc_constraint — it mixes contact forces with friction.
    """
    import mujoco

    dof_idx = _arm_dof_indices(robot)

    # ---- tau_motor -------------------------------------------------------
    tau_motor = np.array(data.qfrc_actuator)[dof_idx].astype(np.float64)

    # ---- tau_gravity (zero-velocity trick) -------------------------------
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    tau_gravity = np.array(scratch.qfrc_bias)[dof_idx].astype(np.float64)

    # ---- tau_coriolis ----------------------------------------------------
    tau_bias     = np.array(data.qfrc_bias)[dof_idx].astype(np.float64)
    tau_coriolis = tau_bias - tau_gravity

    # ---- tau_inertial  (M(q) @ qacc) -------------------------------------
    nv = model.nv
    M = np.zeros((nv, nv), dtype=np.float64)
    try:
        mujoco.mj_fullM(model, data, M)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)
    tau_inertial = (M @ np.array(data.qacc))[dof_idx].astype(np.float64)

    # ---- tau_friction  (reconstruct, do NOT use qfrc_constraint) ---------
    vel  = np.array(data.qvel)[dof_idx].astype(np.float64)
    fl   = np.array(model.dof_frictionloss)[dof_idx].astype(np.float64)
    damp = np.array(model.dof_damping)[dof_idx].astype(np.float64)
    # Coulomb: -sign(v) * frictionloss  |  viscous: -damping * v
    tau_friction = -(np.sign(vel) * fl + damp * vel)

    # ---- tau_model  (model-known balance demand) -------------------------
    tau_model = tau_gravity + tau_coriolis + tau_inertial + tau_friction

    return {
        "tau_motor":    tau_motor,
        "tau_gravity":  tau_gravity,
        "tau_coriolis": tau_coriolis,
        "tau_inertial": tau_inertial,
        "tau_friction": tau_friction,
        "tau_model":    tau_model,
    }
