"""
get_ground_truth_dynamics(model, data, robot, *, grouped=False, load_body=None) -> dict

Composes the two ground-truth groups for the SOARM101 from a live MuJoCo sim:

    ground_truth.model.*  (model_terms.py) — rigid-body torque decomposition
        tau_motor, tau_gravity, tau_coriolis, tau_inertial, tau_friction, tau_model
    ground_truth.ext.*    (external.py)    — external-force truth
        tau_ext, tau_ext_residual, tau_ext_jac, tcp_wrench_ext, tcp_wrench_ft

Call per control step (or per physics substep via SOARM101Sim.substep_hook).

Return shape:
    grouped=True  -> {"model": {...}, "ext": {...}}
    grouped=False -> flat dict (backward compatible).  The legacy key
        "tcp_wrench" is preserved and equals tcp_wrench_ft (raw FT sensor,
        wrist body frame); the world-frame external wrench is "tcp_wrench_ext".

The shared equation of motion is  M·q̈ = τ_motor − (G + C·q̇) − friction − τ_ext.
See model_terms.py / external.py for per-term definitions and sign conventions.
"""

from __future__ import annotations

from robosuite_private.dynamics_gt.external import compute_external_terms
from robosuite_private.dynamics_gt.model_terms import _arm_dof_indices, compute_model_terms

__all__ = ["get_ground_truth_dynamics", "_arm_dof_indices"]


def get_ground_truth_dynamics(model, data, robot, *, grouped: bool = False, load_body=None) -> dict:
    """
    Compute the full ground-truth dynamics decomposition for the SOARM101.

    Args:
        model: mujoco.MjModel (from env.sim.model._model)
        data:  mujoco.MjData  (from env.sim.data._data)
        robot: robosuite robot instance (env.robots[0])
        grouped: if True, return {"model": {...}, "ext": {...}}; else a flat dict.
        load_body: optional body name carrying a known applied wrench (forwarded
            to compute_external_terms).  If None, auto-detect from xfrc_applied.

    Returns:
        dict — grouped or flat per the `grouped` flag.
    """
    model_terms = compute_model_terms(model, data, robot)
    ext_terms = compute_external_terms(
        model, data, robot, load_body=load_body, model_terms=model_terms
    )

    if grouped:
        return {"model": model_terms, "ext": ext_terms}

    # ---- flat / backward-compatible view --------------------------------
    flat = dict(model_terms)            # tau_motor, tau_gravity, tau_coriolis,
                                        # tau_inertial, tau_friction, tau_model
    flat["tau_ext"] = ext_terms["tau_ext"]
    flat["tau_ext_residual"] = ext_terms["tau_ext_residual"]
    flat["tau_ext_jac"] = ext_terms["tau_ext_jac"]
    flat["tau_ext_contact"] = ext_terms["tau_ext_contact"]
    flat["tcp_wrench_ext"] = ext_terms["tcp_wrench_ext"]
    # Legacy key: "tcp_wrench" historically meant the raw FT sensor reading.
    flat["tcp_wrench"] = ext_terms["tcp_wrench_ft"]
    flat["tcp_wrench_ft"] = ext_terms["tcp_wrench_ft"]
    return flat
