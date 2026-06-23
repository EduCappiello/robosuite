"""
body_jacobian(model, data, body_id, dof_idx) -> (6, len(dof_idx))

Spatial Jacobian of a body's center of mass, sliced to the requested DOF
columns.  Used by external.py to map an external Cartesian wrench to joint
torques (τ = Jᵀ·w).

The center-of-mass reference (mj_jacBodyCom, point = data.xipos[body]) is
chosen deliberately: MuJoCo's xfrc_applied acts at the body COM in the world
frame, so a COM Jacobian makes  Jᵀ·xfrc_applied  equal the generalized force
MuJoCo itself applies.  Using the body-frame-origin Jacobian (mj_jacBody) would
introduce a moment-arm error equal to the COM offset (risk R7).

Row order is [translation(3); rotation(3)] so it pairs with a wrench laid out
as [fx,fy,fz, tx,ty,tz] (force then torque), matching xfrc_applied.
"""

from __future__ import annotations

import numpy as np


def body_jacobian(model, data, body_id: int, dof_idx) -> np.ndarray:
    """
    Return the (6, k) COM spatial Jacobian of `body_id` for the columns in
    `dof_idx` (k = len(dof_idx)).

    Args:
        model:   mujoco.MjModel
        data:    mujoco.MjData (must be at the desired state; caller runs
                 mj_forward/step beforehand)
        body_id: target body index (e.g. model.site_bodyid[eef_site_id])
        dof_idx: iterable of nv-space DOF indices to keep as columns.

    Returns:
        (6, k) float64 array.  Rows 0:3 translational (jacp), rows 3:6
        rotational (jacr).  J^T @ [f; t] gives the joint torques produced by a
        world-frame wrench [f; t] applied at the body COM.
    """
    import mujoco

    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, int(body_id))

    dof_idx = list(dof_idx)
    J = np.vstack([jacp, jacr])           # (6, nv): [translation; rotation]
    return J[:, dof_idx].astype(np.float64)
