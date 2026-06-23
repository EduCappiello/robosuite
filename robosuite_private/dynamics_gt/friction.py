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
