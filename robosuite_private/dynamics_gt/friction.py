"""
set_friction_params(model, coeffs_by_joint) -> None

Writes per-joint Coulomb and viscous friction coefficients into the MuJoCo
model.  Accepts raw numbers only — does NOT load calibration files (that
responsibility stays in lerobot-private).

The updated coefficients are immediately reflected by get_ground_truth_dynamics
because ground_truth.py reads them live from model.dof_frictionloss / dof_damping.
"""

from __future__ import annotations


def set_friction_params(model, coeffs_by_joint: dict) -> None:
    """
    Update friction coefficients in the MuJoCo model for named joints.

    Args:
        model: mujoco.MjModel (mutated in place).
        coeffs_by_joint: mapping of joint_name -> dict with any subset of:
            "frictionloss"  (float) — Coulomb friction [N·m]
            "damping"       (float) — viscous damping  [N·m·s/rad]
            "armature"      (float) — rotor inertia    [kg·m²]

    Example::

        set_friction_params(model, {
            "shoulder_pan":  {"frictionloss": 0.06, "damping": 0.65},
            "shoulder_lift": {"frictionloss": 0.05, "damping": 0.60},
        })
    """
    import mujoco

    for joint_name, coeffs in coeffs_by_joint.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        # When loaded via robosuite the model prefixes joints (e.g. "robot0_shoulder_pan").
        # Try the prefixed form if the bare name is not found.
        if jnt_id < 0:
            prefixed = f"robot0_{joint_name}"
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
        if jnt_id < 0:
            raise ValueError(
                f"Joint '{joint_name}' not found in model "
                f"(tried bare name and 'robot0_{joint_name}')."
            )
        dof_adr = int(model.jnt_dofadr[jnt_id])

        if "frictionloss" in coeffs:
            model.dof_frictionloss[dof_adr] = float(coeffs["frictionloss"])
        if "damping" in coeffs:
            model.dof_damping[dof_adr] = float(coeffs["damping"])
        if "armature" in coeffs:
            model.dof_armature[dof_adr] = float(coeffs["armature"])


def apply_load_dependent_limit(model, data, entries, static_eps: float = 0.02) -> None:
    """Set each dof's Coulomb breakaway limit from the CURRENT load, using the right kappa.

    MuJoCo's dof_frictionloss is a genuine static breakaway limit, not a kinetic term: a
    joint holding 0.099 N*m of gravity with zero actuator does not slide once the limit is
    above it (verified, scripts_tmp/probe_stiction_exists.py). So the limit IS the stiction
    model, and it must be a property maintained on the model rather than a side effect of
    one particular control loop -- otherwise any other evaluation path (ground truth, a
    probe, a replay) silently sees the XML default of ~0.05 N*m, which is 25x too small for
    a loaded arm.

    The real arm's static and kinetic load couplings are different physical quantities and
    were fitted separately (next_free_ekf_live.json carries both `kappa` and
    `kappa_static`). Using the kinetic kappa at zero velocity under-predicts the holding
    torque the gearbox actually supplies. Pick by |qdot|:

      |qdot| < static_eps  ->  Fc + kappa_static * |G|     (stiction branch)
      otherwise            ->  Fc + kappa_kinetic * |G|    (sliding branch)

    G is qfrc_bias for that dof (gravity + Coriolis), the same load proxy the real
    calibration uses.

    Args:
        entries: iterable of (dof_adr, Fc, kappa_kinetic, kappa_static).
    """
    import numpy as np

    bias = data.qfrc_bias
    vel = data.qvel
    for dof, fc, kap_kin, kap_stat in entries:
        kap = kap_stat if abs(float(vel[dof])) < static_eps else kap_kin
        model.dof_frictionloss[dof] = float(fc) + float(kap) * abs(float(bias[dof]))
