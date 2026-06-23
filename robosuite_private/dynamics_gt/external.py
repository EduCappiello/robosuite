"""
compute_external_terms(model, data, robot, *, load_body=None) -> dict
    ("ground_truth.ext.*")  + apply_external_load / clear_external_loads

External-force ground truth: the external joint torque τ_ext and external TCP
wrench that lerobot-private's estimators try to recover from motor signals.

Sign convention (matches the estimator's tau_ext = tau_motor - tau_model and the
shared EoM  M·q̈ = τ_motor − (G + C·q̇) − friction − τ_ext):

    τ_ext is the external torque the motor must work AGAINST.  A world-frame
    wrench w applied to a body produces the generalized force  J_comᵀ·w  that
    *adds* to the dynamics, so in this convention

        τ_ext = − J_comᵀ · w_ext.

Verified identity (proved in §JOURNAL): with Coulomb frictionloss = 0, after a
single mj_forward,  tau_ext_residual == tau_ext_jac  EXACTLY (viscous damping
cancels because the reconstructed −damp·v equals MuJoCo's qfrc_passive).

Schema (DOF order = 5 arm joints then gripper):
    tau_ext          (6,) — primary external joint torque.  Jacobian truth when a
                            known wrench is present, else the model residual.
    tau_ext_residual (6,) — analytic EoM residual; the quantity the estimator
                            itself computes from motor signals (always present).
    tau_ext_jac      (6,) — −Jᵀ·w from a known applied wrench, else zeros.
    tcp_wrench_ext   (6,) — purely-external wrench at the TCP, WORLD frame
                            [fx,fy,fz,tx,ty,tz] (physical applied wrench), else zeros.
    tcp_wrench_ft    (6,) — raw wrist FT sensor (total transmitted, gripper body
                            frame).  Diagnostic only — never fed into tau_ext.
"""

from __future__ import annotations

import numpy as np

from robosuite_private.dynamics_gt.jacobian import body_jacobian
from robosuite_private.dynamics_gt.model_terms import _arm_dof_indices, compute_model_terms

_NONZERO_WRENCH_TOL = 1e-12


def _robot_body_ids(model, robot) -> set:
    """Body ids belonging to the robot (its naming-prefixed bodies, e.g. ``robot0_*``)."""
    import mujoco

    prefix = robot.robot_model.naming_prefix
    out = set()
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(prefix):
            out.add(b)
    return out


def compute_contact_ext_torque(model, data, robot, *, dof_idx=None, return_contacts=False):
    """External joint torque from the sim's ACTUAL contacts — robot↔world only.

    Unlike ``tau_ext_residual`` (τ_motor − τ_model, which absorbs friction, joint limits,
    actuator saturation and self-collision), this reads the real contact forces directly:
    it is EXACTLY zero in free space regardless of controller state, ignores robot↔robot
    self-collisions, and equals the true reaction when the arm/gripper actually touches the
    table, the cube, or a grasped object. That makes it the clean signal for force feedback.

    Sign matches ``tau_ext_residual`` (τ_ext = −Jᵀ·w_ext, the torque the motor works against),
    so the two coincide when the only external load is a real contact.

    Returns ``tau`` (DOF-ordered, arm then gripper), or ``(tau, contacts)`` when
    ``return_contacts`` — ``contacts`` is a list of ``(geom_a, geom_b, |force| N)`` for each
    external contact (handy for spotting which collision geom, e.g. ``static_finger_tip``, is hitting).
    """
    import mujoco

    if dof_idx is None:
        dof_idx = _arm_dof_indices(robot)
    robot_bodies = _robot_body_ids(model, robot)
    prefix = robot.robot_model.naming_prefix

    tau = np.zeros(model.nv, dtype=np.float64)
    contacts = []
    f6 = np.zeros(6, dtype=np.float64)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        r1, r2 = b1 in robot_bodies, b2 in robot_bodies
        if r1 == r2:
            continue  # robot↔robot (self) or world↔world — not an external robot contact

        mujoco.mj_contactForce(model, data, i, f6)         # 6D [force, torque], contact frame
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame.T @ f6[:3]                     # contact-frame force → world
        torque_world = frame.T @ f6[3:6]                   # torsional/rolling moment (condim>3) → world
        # mj_contactForce gives the force on geom1 from geom2 (acting along +normal away from
        # geom2). Apply it to the robot body: + when the robot owns geom1, − when it owns geom2.
        robot_body = b1 if r1 else b2
        sign = 1.0 if r1 else -1.0
        mujoco.mj_jac(model, data, jacp, jacr, np.array(c.pos, dtype=np.float64), robot_body)
        tau += sign * (jacp.T @ force_world + jacr.T @ torque_world)

        if return_contacts:
            g1 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1))
            g2 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2))
            contacts.append((g1.replace(prefix, ""), g2.replace(prefix, ""),
                             float(np.linalg.norm(force_world))))

    # Sign chosen to match tau_ext_residual (the motor works AGAINST the contact reaction),
    # verified against the residual on a clean single contact.
    tau_dof = tau[dof_idx].astype(np.float64)
    return (tau_dof, contacts) if return_contacts else tau_dof


def _resolve_body_id(model, body_name: str) -> int:
    """Resolve a body name to its id, trying the robot0_ prefix as a fallback."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot0_{body_name}")
    return bid


def _find_applied_wrench(model, data, load_body):
    """
    Locate the body carrying an external wrench in data.xfrc_applied.

    Returns (body_id, w_world (6,), p_app (3,)) or (None, None, None) if there is
    no known applied wrench.  xfrc_applied is [fx,fy,fz,tx,ty,tz] in the world
    frame, applied at the body center of mass (data.xipos).
    """
    xfrc = np.array(data.xfrc_applied, dtype=np.float64)  # (nbody, 6)

    if load_body is not None:
        bid = _resolve_body_id(model, load_body)
        if bid < 0:
            raise ValueError(
                f"Body '{load_body}' not found (tried bare name and 'robot0_{load_body}')."
            )
    else:
        nz = np.where(np.any(np.abs(xfrc) > _NONZERO_WRENCH_TOL, axis=1))[0]
        if len(nz) == 0:
            return None, None, None
        bid = int(nz[0])  # first body with a nonzero applied wrench

    w_world = xfrc[bid].copy()
    p_app = np.array(data.xipos[bid], dtype=np.float64)  # COM, where xfrc acts
    return bid, w_world, p_app


def _read_ft_sensor(robot) -> np.ndarray:
    """Raw wrist FT sensor wrench [F(3), M(3)] in the wrist body frame, or zeros."""
    prefix = robot.robot_model.naming_prefix  # e.g. "robot0_"
    try:
        ft_force = robot.get_sensor_measurement(f"{prefix}wrist_ft_force")
        ft_torque = robot.get_sensor_measurement(f"{prefix}wrist_ft_torque")
        return np.concatenate([ft_force, ft_torque]).astype(np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


def compute_external_terms(model, data, robot, *, load_body=None, model_terms=None) -> dict:
    """
    Compute the external-force ground truth (ground_truth.ext.*).

    Args:
        model: mujoco.MjModel
        data:  mujoco.MjData
        robot: robosuite robot instance (env.robots[0])
        load_body: optional body name carrying the known applied wrench.  If
                   None, auto-detect the first body with a nonzero xfrc_applied.
        model_terms: optional precomputed compute_model_terms() dict (avoids a
                   redundant recompute when called from the aggregator).

    Returns:
        dict matching the schema above.
    """
    dof_idx = _arm_dof_indices(robot)

    if model_terms is None:
        model_terms = compute_model_terms(model, data, robot)
    tau_motor = model_terms["tau_motor"]
    tau_gravity = model_terms["tau_gravity"]
    tau_coriolis = model_terms["tau_coriolis"]
    tau_inertial = model_terms["tau_inertial"]
    tau_friction = model_terms["tau_friction"]

    # ---- tau_ext_residual (R1) ------------------------------------------
    # τ_ext = τ_motor − G − C − M·q̈ + tau_friction
    # The trailing + is because tau_friction is stored in APPLIED (negative)
    # form; physically the motor works against +friction.  This expression is
    # exactly what the estimator computes as tau_motor − tau_model.
    tau_ext_residual = (
        tau_motor - tau_gravity - tau_coriolis - tau_inertial + tau_friction
    ).astype(np.float64)

    # ---- locate a known applied wrench ----------------------------------
    body_id, w_world, p_app = _find_applied_wrench(model, data, load_body)

    # ---- tau_ext_jac = −Jᵀ·w  (negated to match the residual convention) -
    if body_id is not None:
        J = body_jacobian(model, data, body_id, dof_idx)  # (6, 6)
        tau_ext_jac = (-(J.T @ w_world)).astype(np.float64)
    else:
        tau_ext_jac = np.zeros(6, dtype=np.float64)

    # ---- primary tau_ext: Jacobian truth if a load is known, else residual
    tau_ext = tau_ext_jac if body_id is not None else tau_ext_residual

    # ---- tcp_wrench_ext: physical applied wrench transported to the TCP --
    p_tcp = np.array(data.site_xpos[robot.eef_site_id["right"]], dtype=np.float64)
    if body_id is not None:
        f = w_world[:3]
        t = w_world[3:]
        r = p_app - p_tcp  # from TCP to application point
        t_tcp = t + np.cross(r, f)
        tcp_wrench_ext = np.concatenate([f, t_tcp]).astype(np.float64)
    else:
        tcp_wrench_ext = np.zeros(6, dtype=np.float64)

    # ---- tcp_wrench_ft: raw FT sensor (diagnostic only) -----------------
    tcp_wrench_ft = _read_ft_sensor(robot)

    # ---- tau_ext_contact: from the sim's ACTUAL external contacts -------
    # Clean FF signal: 0 in free space, real reaction on table/cube/grasp contact, ignores
    # self-collision and is immune to the controller saturation/friction that pollute the residual.
    tau_ext_contact = compute_contact_ext_torque(model, data, robot, dof_idx=dof_idx)

    return {
        "tau_ext": tau_ext,
        "tau_ext_residual": tau_ext_residual,
        "tau_ext_jac": tau_ext_jac,
        "tau_ext_contact": tau_ext_contact,
        "tcp_wrench_ext": tcp_wrench_ext,
        "tcp_wrench_ft": tcp_wrench_ft,
    }


# ---------------------------------------------------------------------------
# Known-load injection helpers (mutate the sim; used by tests/harness, NOT by
# the read path).  Caller is responsible for running mj_forward / stepping.
# ---------------------------------------------------------------------------


def apply_external_load(model, data, body_name: str, wrench) -> int:
    """
    Impose a known external wrench on a body via data.xfrc_applied.

    Args:
        model: mujoco.MjModel
        data:  mujoco.MjData (mutated in place)
        body_name: target body (bare or robot0_-prefixed).
        wrench: length-6 [fx,fy,fz,tx,ty,tz], world frame, applied at the body COM.

    Returns:
        The resolved body id.
    """
    bid = _resolve_body_id(model, body_name)
    if bid < 0:
        raise ValueError(
            f"Body '{body_name}' not found (tried bare name and 'robot0_{body_name}')."
        )
    w = np.asarray(wrench, dtype=np.float64).reshape(6)
    data.xfrc_applied[bid, :] = w
    return bid


def clear_external_loads(data) -> None:
    """Zero all applied external wrenches (data.xfrc_applied[:] = 0)."""
    data.xfrc_applied[:] = 0.0
