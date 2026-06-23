"""
Verification tests for the external-force ground truth (ground_truth.ext.*):
compute_external_terms, apply_external_load / clear_external_loads.

Core identity (T8): with Coulomb frictionloss = 0, MuJoCo's own force balance
guarantees, after a single mj_forward at a contact-free pose,

    tau_ext_residual  ==  tau_ext_jac  ==  −Jᵀ_com · w_ext

(the viscous damping in the reconstructed friction cancels MuJoCo's qfrc_passive).
This is the bridge between the analytic EoM residual the estimator computes and
the Jacobian truth from a known applied wrench.
"""

import contextlib

import numpy as np
import pytest

import robosuite as suite
from robosuite_private.dynamics_gt import (
    apply_external_load,
    clear_external_loads,
    compute_external_terms,
    compute_model_terms,
)

_MODEL_KEYS = {
    "tau_motor", "tau_gravity", "tau_coriolis",
    "tau_inertial", "tau_friction", "tau_model",
}
_EXT_KEYS = {
    "tau_ext", "tau_ext_residual", "tau_ext_jac", "tau_ext_contact",
    "tcp_wrench_ext", "tcp_wrench_ft",
}


@pytest.fixture(scope="module")
def env_robot():
    env = suite.make(
        env_name="Lift",
        robots=["SOARM101"],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    robot = env.robots[0]
    yield env, robot
    env.close()


@contextlib.contextmanager
def _zero_frictionloss(model):
    """Temporarily zero Coulomb frictionloss on all DOFs (viscous damping kept)."""
    saved = model.dof_frictionloss.copy()
    model.dof_frictionloss[:] = 0.0
    try:
        yield
    finally:
        model.dof_frictionloss[:] = saved


def _eef_body(env, robot):
    """(model, data, eef_body_id, eef_body_name) for the body owning the grip site."""
    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    bid = int(model.site_bodyid[robot.eef_site_id["right"]])
    name = mujoco.mj_name2id  # silence linters; real call below
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return model, data, bid, bname


# --------------------------------------------------------------------------- T1/T2

def test_external_schema(env_robot):
    env, robot = env_robot
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    ext = compute_external_terms(model, data, robot)
    assert set(ext.keys()) == _EXT_KEYS
    for k, arr in ext.items():
        assert isinstance(arr, np.ndarray), f"{k} must be ndarray"
        assert arr.shape == (6,), f"{k} shape {arr.shape} != (6,)"
        assert arr.dtype == np.float64, f"{k} dtype {arr.dtype} != float64"


def test_model_schema(env_robot):
    env, robot = env_robot
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    mt = compute_model_terms(model, data, robot)
    assert set(mt.keys()) == _MODEL_KEYS
    for k, arr in mt.items():
        assert arr.shape == (6,) and arr.dtype == np.float64, f"{k} bad shape/dtype"


def test_tau_model_additivity(env_robot):
    """tau_model == gravity + coriolis + inertial + friction (T5)."""
    env, robot = env_robot
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    mt = compute_model_terms(model, data, robot)
    expected = mt["tau_gravity"] + mt["tau_coriolis"] + mt["tau_inertial"] + mt["tau_friction"]
    np.testing.assert_allclose(mt["tau_model"], expected, atol=1e-12)


# --------------------------------------------------------------------------- T7

def test_no_load_residual_zero(env_robot):
    """No applied wrench → tau_ext_jac is zero and (zero-friction) residual ≈ 0 (T7)."""
    import mujoco

    env, robot = env_robot
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    clear_external_loads(data)
    with _zero_frictionloss(model):
        mujoco.mj_forward(model, data)
        ext = compute_external_terms(model, data, robot)
    np.testing.assert_allclose(ext["tau_ext_jac"], np.zeros(6), atol=1e-12)
    # With no external force, the EoM residual is the zero vector (identity).
    np.testing.assert_allclose(ext["tau_ext_residual"], np.zeros(6), atol=1e-6)
    # tau_ext falls back to the residual when no load is known.
    np.testing.assert_allclose(ext["tau_ext"], ext["tau_ext_residual"], atol=1e-12)


# --------------------------------------------------------------------------- T8 (core)

def test_jac_equals_residual_under_known_load(env_robot):
    """Zero-friction quasi-static: tau_ext_jac == tau_ext_residual for a known load (T8)."""
    import mujoco

    env, robot = env_robot
    env.reset()
    model, data, bid, bname = _eef_body(env, robot)

    with _zero_frictionloss(model):
        # Lateral + downward force at the eef body COM (pure force, no torque).
        wrench = np.array([1.0, 0.0, -2.0, 0.0, 0.0, 0.0], dtype=np.float64)
        apply_external_load(model, data, bname, wrench)
        mujoco.mj_forward(model, data)
        ext = compute_external_terms(model, data, robot, load_body=bname)
        clear_external_loads(data)

    np.testing.assert_allclose(ext["tau_ext_jac"], ext["tau_ext_residual"], atol=1e-5)
    # When a load is known, the primary tau_ext is the Jacobian truth.
    np.testing.assert_allclose(ext["tau_ext"], ext["tau_ext_jac"], atol=1e-12)
    # A wrench on the eef body produces no torque about the distal gripper DOF.
    assert abs(ext["tau_ext_jac"][5]) < 1e-6, "gripper DOF column should be ~0 (R4)"


# --------------------------------------------------------------------------- T9 (guards R1)

def test_residual_matches_raw_formula(env_robot):
    """tau_ext_residual == motor − gravity − coriolis − inertial + friction (R1 sign)."""
    import mujoco

    env, robot = env_robot
    env.reset()
    model, data, bid, bname = _eef_body(env, robot)
    apply_external_load(model, data, bname, np.array([0, 0, -1.5, 0, 0, 0], float))
    mujoco.mj_forward(model, data)

    mt = compute_model_terms(model, data, robot)
    ext = compute_external_terms(model, data, robot, load_body=bname, model_terms=mt)
    expected = (
        mt["tau_motor"] - mt["tau_gravity"] - mt["tau_coriolis"]
        - mt["tau_inertial"] + mt["tau_friction"]
    )
    np.testing.assert_allclose(ext["tau_ext_residual"], expected, atol=1e-12)
    clear_external_loads(data)


# --------------------------------------------------------------------------- T10 (guards R2)

def test_tcp_wrench_transport(env_robot):
    """tcp_wrench_ext recovers the imposed world force and its transported moment (R2)."""
    import mujoco

    env, robot = env_robot
    env.reset()
    model, data, bid, bname = _eef_body(env, robot)

    f = np.array([0.5, -0.8, -2.0], dtype=np.float64)
    apply_external_load(model, data, bname, np.concatenate([f, np.zeros(3)]))
    mujoco.mj_forward(model, data)
    ext = compute_external_terms(model, data, robot, load_body=bname)

    p_app = np.array(data.xipos[bid], dtype=np.float64)
    p_tcp = np.array(data.site_xpos[robot.eef_site_id["right"]], dtype=np.float64)
    expected_moment = np.cross(p_app - p_tcp, f)

    np.testing.assert_allclose(ext["tcp_wrench_ext"][:3], f, atol=1e-9)
    np.testing.assert_allclose(ext["tcp_wrench_ext"][3:], expected_moment, atol=1e-9)
    clear_external_loads(data)


# --------------------------------------------------------------------------- T11

def test_ft_sensor_vs_external_distinction(env_robot):
    """Under gravity-only: tcp_wrench_ext == 0 but the raw FT sensor reads nonzero (T11)."""
    import mujoco

    env, robot = env_robot
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    clear_external_loads(data)
    mujoco.mj_forward(model, data)
    ext = compute_external_terms(model, data, robot)

    np.testing.assert_allclose(ext["tcp_wrench_ext"], np.zeros(6), atol=1e-12)
    # The FT sensor transmits distal gravity load → nonzero even with no external wrench.
    assert np.linalg.norm(ext["tcp_wrench_ft"]) > 1e-3, (
        "FT sensor should read the distal gravity load"
    )


# --------------------------------------------------------------------------- helpers

def test_apply_and_clear_roundtrip(env_robot):
    """apply_external_load sets xfrc on the named body; clear_external_loads zeros it."""
    env, robot = env_robot
    env.reset()
    model, data, bid, bname = _eef_body(env, robot)

    w = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    returned = apply_external_load(model, data, bname, w)
    assert returned == bid
    np.testing.assert_allclose(np.array(data.xfrc_applied[bid]), w)

    clear_external_loads(data)
    assert np.all(np.array(data.xfrc_applied) == 0.0)
