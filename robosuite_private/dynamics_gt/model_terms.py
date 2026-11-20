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
    tau_friction — friction the solver actually applied (Coulomb/static from the constraint
                   rows) plus viscous damping. APPLIED sign, negative when it opposes motion.
                   At a static hold this is NON-ZERO up to the breakaway limit.
    tau_model    — tau_gravity + tau_coriolis + tau_inertial + tau_friction
                   (matches lerobot's TorqueEstimate.tau_model)
"""

from __future__ import annotations

import numpy as np

# mj_fullM's signature changed across mujoco builds. Probing it with try/except on every
# call is ruinously expensive: the failing call raises a pybind11 TypeError whose message
# repr()s the whole mass matrix. Resolve the signature at most once per process and cache it
# (same fix as robosuite/controllers/parts/controller.py::_compute_full_mass_matrix).
_MJ_FULLM_TAKES_DATA = None


def _fullM(model, data, dst) -> None:
    """Fill `dst` with the dense mass matrix using this build's mj_fullM signature."""
    global _MJ_FULLM_TAKES_DATA
    import mujoco
    if _MJ_FULLM_TAKES_DATA is None:
        try:
            mujoco.mj_fullM(model, data, dst)
            _MJ_FULLM_TAKES_DATA = True
            return
        except TypeError:
            _MJ_FULLM_TAKES_DATA = False
    if _MJ_FULLM_TAKES_DATA:
        mujoco.mj_fullM(model, data, dst)
    else:
        mujoco.mj_fullM(model, dst, data.qM)


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
    _fullM(model, data, M)
    tau_inertial = (M @ np.array(data.qacc))[dof_idx].astype(np.float64)

    # ---- tau_friction  (read what the solver APPLIED, do not reconstruct) ---
    #
    # *** SEMANTICS CHANGE (Eduardo, 2026-11-19: keep the fix, flag it) ***
    # This field now reports the friction force the CONSTRAINT SOLVER ACTUALLY APPLIED,
    # which is non-zero at a static hold (up to the breakaway limit). It used to be an
    # analytic reconstruction, -(sign(v)*frictionloss + damping*v), which is identically 0
    # at v = 0. Anything that consumed the old meaning -- e.g. a D_F / force-profile
    # comparison, or a residual check that assumed tau_friction == 0 at rest -- will see
    # different numbers from this change alone. Consumers that need the old quantity can
    # reconstruct it from model.dof_frictionloss and data.qvel directly.
    #
    # The previous form, -(sign(v)*frictionloss + damping*v), is 0 at v = 0. That is not a
    # modelling choice, it is an artifact of sign(0) = 0, and it hides friction the physics is
    # really applying: MuJoCo's dof_frictionloss is a genuine STATIC breakaway limit, verified
    # directly (scripts_tmp/probe_stiction_exists.py) -- a joint holding 0.099 N*m of gravity
    # with ZERO actuator does not slide once frictionloss >= ~0.15, and free-falls at 0. So at
    # a static hold the old term reported ~0 while the solver was supplying up to the limit.
    # That made the sim look 5-7x friction-light against the real estimator, whose static gate
    # does report the holding torque -- an apples-to-oranges comparison, not a model gap.
    #
    # MuJoCo exposes the applied force per constraint row. The dof-frictionloss rows are
    # mjtConstraint FRICTIONDOF (enum value 1; this binding exposes no mjCON_* constants), with
    # efc_id = dof index and efc_frictionloss = the limit. Verified in
    # scripts_tmp/probe_efc_friction.py: 6 rows, efc_id = [0..5], efc_force = the joint torque.
    # Viscous damping is not in those rows, so it stays analytic (MuJoCo computes it exactly).
    vel  = np.array(data.qvel)[dof_idx].astype(np.float64)
    fl   = np.array(model.dof_frictionloss)[dof_idx].astype(np.float64)
    damp = np.array(model.dof_damping)[dof_idx].astype(np.float64)

    coulomb = -(np.sign(vel) * fl)          # fallback: analytic, only valid while moving
    nefc = int(data.nefc)
    if nefc > 0:
        efc_type = np.asarray(data.efc_type)[:nefc]
        rows = np.flatnonzero(efc_type == 1)          # 1 = FRICTIONDOF
        if rows.size:
            efc_id = np.asarray(data.efc_id)[rows]
            efc_force = np.asarray(data.efc_force)[rows]
            pos = {int(dd): k for k, dd in enumerate(dof_idx)}
            for i, dd in enumerate(efc_id):        # efc_force[i] pairs with rows[i]
                k = pos.get(int(dd))
                if k is not None:
                    coulomb[k] = float(efc_force[i])
    tau_friction = coulomb - damp * vel

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
