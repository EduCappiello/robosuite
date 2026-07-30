"""
Per-arm ground-truth verification on the 17-DoF XLeRobot (M2 of the sim port).

Extends the T7/T8 identity tests of test_external_gt.py to the bimanual mobile
platform: each arm's (6,) decomposition must satisfy the free-space and
known-load identities independently, and a wrench applied to ONE arm must not
leak into the other arm's ground truth (cross-arm isolation — the arms only
share ancestors through the chassis/base joints, which are not arm DOFs).
"""

import contextlib

import numpy as np
import pytest

from robosuite_private.dynamics_gt import (
    apply_external_load,
    clear_external_loads,
    compute_external_terms,
    get_ground_truth_dynamics,
)
from robosuite_private.motor_signals import read_motor_signals
from robosuite_private.sim_api import XLeRobotSim

ARMS = ("right", "left")
_MODEL_KEYS = {
    "tau_motor", "tau_gravity", "tau_coriolis",
    "tau_inertial", "tau_friction", "tau_model",
}
_EXT_KEYS = {
    "tau_ext", "tau_ext_residual", "tau_ext_jac", "tau_ext_contact",
    "tcp_wrench_ext", "tcp_wrench_ft",
}


@pytest.fixture(scope="module")
def sim():
    s = XLeRobotSim()
    s.reset()
    yield s
    s.close()


@contextlib.contextmanager
def _zero_frictionloss(model):
    """Temporarily zero Coulomb frictionloss on all DOFs (viscous damping kept)."""
    saved = model.dof_frictionloss.copy()
    model.dof_frictionloss[:] = 0.0
    try:
        yield
    finally:
        model.dof_frictionloss[:] = saved


def _md(sim):
    return sim.env.sim.model._model, sim.env.sim.data._data


def test_per_arm_schema(sim):
    sim.reset()
    model, data = _md(sim)
    out = get_ground_truth_dynamics(model, data, sim.robot, grouped=True, arms=ARMS)
    assert set(out.keys()) == set(ARMS)
    for arm in ARMS:
        assert set(out[arm]["model"].keys()) == _MODEL_KEYS
        assert set(out[arm]["ext"].keys()) == _EXT_KEYS
        for k, arr in {**out[arm]["model"], **out[arm]["ext"]}.items():
            assert arr.shape == (6,) and arr.dtype == np.float64, f"{arm}.{k}"


def test_free_space_zero_both_arms(sim):
    """No load, no contact on either arm → per-arm residual ≈ 0, jac = 0, contact = 0."""
    import mujoco

    sim.reset()
    model, data = _md(sim)
    clear_external_loads(data)
    with _zero_frictionloss(model):
        mujoco.mj_forward(model, data)
        for arm in ARMS:
            ext = compute_external_terms(model, data, sim.robot, arm=arm)
            np.testing.assert_allclose(ext["tau_ext_jac"], np.zeros(6), atol=1e-12,
                                       err_msg=f"{arm}: jac")
            np.testing.assert_allclose(ext["tau_ext_contact"], np.zeros(6), atol=1e-9,
                                       err_msg=f"{arm}: contact")
            np.testing.assert_allclose(ext["tau_ext_residual"], np.zeros(6), atol=1e-6,
                                       err_msg=f"{arm}: residual")


@pytest.mark.parametrize("loaded,other", [("left", "right"), ("right", "left")])
def test_known_load_isolates_to_loaded_arm(sim, loaded, other):
    """T8 per arm + cross-arm isolation: jac == residual on the loaded arm; the
    unloaded arm sees exactly nothing (its DOFs are not ancestors of the load)."""
    import mujoco

    sim.reset()
    model, data = _md(sim)
    with _zero_frictionloss(model):
        wrench = np.array([1.0, 0.0, -2.0, 0.0, 0.0, 0.0], dtype=np.float64)
        apply_external_load(model, data, f"{loaded}_gripper", wrench)
        mujoco.mj_forward(model, data)
        ext_loaded = compute_external_terms(
            model, data, sim.robot, arm=loaded, load_body=f"{loaded}_gripper"
        )
        ext_other = compute_external_terms(model, data, sim.robot, arm=other)
        clear_external_loads(data)

    # Loaded arm: Jacobian truth == analytic residual, and the load is visible.
    np.testing.assert_allclose(
        ext_loaded["tau_ext_jac"], ext_loaded["tau_ext_residual"], atol=1e-5
    )
    assert np.abs(ext_loaded["tau_ext_jac"][:5]).max() > 0.01
    # A wrench on the eef body puts no torque on the distal gripper DOF (R4).
    assert abs(ext_loaded["tau_ext_jac"][5]) < 1e-6

    # Unloaded arm: zero Jacobian projection AND zero residual.
    np.testing.assert_allclose(ext_other["tau_ext_jac"], np.zeros(6), atol=1e-9)
    np.testing.assert_allclose(ext_other["tau_ext_residual"], np.zeros(6), atol=1e-6)


def test_ft_sensors_read_per_arm(sim):
    """Gravity-only: each arm's raw FT sensor reads its own distal load;
    the purely-external TCP wrench stays zero."""
    import mujoco

    sim.reset()
    model, data = _md(sim)
    clear_external_loads(data)
    mujoco.mj_forward(model, data)
    for arm in ARMS:
        ext = compute_external_terms(model, data, sim.robot, arm=arm)
        np.testing.assert_allclose(ext["tcp_wrench_ext"], np.zeros(6), atol=1e-12)
        assert np.linalg.norm(ext["tcp_wrench_ft"]) > 1e-3, (
            f"{arm}: FT sensor should read the distal gravity load"
        )


def test_motor_signals_per_arm(sim):
    sim.reset()
    model, data = _md(sim)
    sigs = {arm: read_motor_signals(sim.robot, kt=0.5, arm=arm) for arm in ARMS}
    for arm in ARMS:
        for key in ("pos", "vel_hw", "current", "current_signed", "load"):
            assert sigs[arm][key].shape == (6,), f"{arm}.{key}"
    # pos must be each arm's own joints: compare against the model directly.
    for arm in ARMS:
        expected = [
            data.qpos[model.joint(f"robot0_{arm}_{n}").qposadr[0]]
            for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        ]
        np.testing.assert_allclose(sigs[arm]["pos"][:5], expected, atol=1e-12)
    # The two arms start at the same init pose but are distinct signal streams.
    assert sigs["right"]["pos"] is not sigs["left"]["pos"]
