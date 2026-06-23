"""
Schema + sanity tests for get_ground_truth_dynamics and set_friction_params.
"""

import numpy as np
import pytest

import robosuite as suite
from robosuite_private.dynamics_gt import get_ground_truth_dynamics, set_friction_params

_MODEL_KEYS = {
    "tau_motor", "tau_gravity", "tau_coriolis",
    "tau_inertial", "tau_friction", "tau_model",
}
_EXT_KEYS = {
    "tau_ext", "tau_ext_residual", "tau_ext_jac", "tau_ext_contact",
    "tcp_wrench_ext", "tcp_wrench_ft",
}
# Flat view = both groups plus the legacy "tcp_wrench" alias (= tcp_wrench_ft).
_EXPECTED_KEYS = _MODEL_KEYS | _EXT_KEYS | {"tcp_wrench"}


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


def _gt(env_robot):
    env, robot = env_robot
    model = env.sim.model._model
    data  = env.sim.data._data
    return get_ground_truth_dynamics(model, data, robot)


def test_schema_keys(env_robot):
    gt = _gt(env_robot)
    assert set(gt.keys()) == _EXPECTED_KEYS


def test_grouped_view(env_robot):
    """grouped=True splits into model/ext groups; flat carries both + legacy alias."""
    env, robot = env_robot
    model = env.sim.model._model
    data  = env.sim.data._data
    grouped = get_ground_truth_dynamics(model, data, robot, grouped=True)
    assert set(grouped.keys()) == {"model", "ext"}
    assert set(grouped["model"].keys()) == _MODEL_KEYS
    assert set(grouped["ext"].keys()) == _EXT_KEYS

    # The flat view must agree with the grouped view value-for-value.
    flat = get_ground_truth_dynamics(model, data, robot, grouped=False)
    for k in _MODEL_KEYS:
        np.testing.assert_allclose(flat[k], grouped["model"][k])
    for k in _EXT_KEYS:
        np.testing.assert_allclose(flat[k], grouped["ext"][k])
    # Legacy alias equals the raw FT sensor.
    np.testing.assert_allclose(flat["tcp_wrench"], grouped["ext"]["tcp_wrench_ft"])


def test_schema_shapes(env_robot):
    gt = _gt(env_robot)
    for key in _EXPECTED_KEYS:
        arr = gt[key]
        assert isinstance(arr, np.ndarray), f"{key} must be ndarray"
        assert arr.shape == (6,), f"{key} shape should be (6,), got {arr.shape}"
        assert arr.dtype == np.float64, f"{key} dtype should be float64"


def test_gravity_sign_at_home(env_robot):
    """
    At the default home pose the arm is partially raised.  Gravity should
    produce a non-zero torque on shoulder_lift (joint index 1).
    We do NOT check magnitude — just that it is non-zero and finite.
    """
    env, robot = env_robot
    env.reset()
    model = env.sim.model._model
    data  = env.sim.data._data
    gt    = get_ground_truth_dynamics(model, data, robot)
    tau_g = gt["tau_gravity"]
    assert np.all(np.isfinite(tau_g)), "tau_gravity contains non-finite values"
    assert np.any(np.abs(tau_g) > 1e-6), "tau_gravity is all zero at home pose"


def test_friction_injection(env_robot):
    """
    After set_friction_params, model coefficients must reflect the new values,
    and get_ground_truth_dynamics must return the updated friction.
    """
    env, robot = env_robot
    import mujoco
    model = env.sim.model._model
    data  = env.sim.data._data

    new_coeffs = {
        "shoulder_pan":  {"frictionloss": 0.11, "damping": 0.77},
        "shoulder_lift": {"frictionloss": 0.09, "damping": 0.65},
    }
    set_friction_params(model, new_coeffs)

    for jname, c in new_coeffs.items():
        # robosuite-loaded models prefix joints with "robot0_"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_{jname}")
        dof = int(model.jnt_dofadr[jid])
        assert abs(model.dof_frictionloss[dof] - c["frictionloss"]) < 1e-9
        assert abs(model.dof_damping[dof]      - c["damping"])      < 1e-9

    # ground_truth should pick up the new values immediately
    gt = get_ground_truth_dynamics(model, data, robot)
    assert np.all(np.isfinite(gt["tau_friction"]))

    # Restore defaults so other tests are not affected.
    # Pass bare names — set_friction_params handles the robot0_ prefix automatically.
    from robosuite_private.robot_spec import JOINT_DAMPING, JOINT_FRICTIONLOSS
    defaults = {j: {"frictionloss": JOINT_FRICTIONLOSS, "damping": JOINT_DAMPING}
                for j in new_coeffs}
    set_friction_params(model, defaults)
