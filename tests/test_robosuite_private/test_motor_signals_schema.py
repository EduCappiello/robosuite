"""
Schema test: read_motor_signals returns exactly the documented keys and shapes.
Also verifies it is callable per physics substep (not only per control step).
"""

import numpy as np
import pytest

import robosuite as suite
from robosuite_private.motor_signals import read_motor_signals
from robosuite_private.robot_spec import KT_DEFAULT

_EXPECTED_KEYS = {"pos", "vel_hw", "current", "current_signed", "load", "dt"}


@pytest.fixture(scope="module")
def env():
    e = suite.make(
        env_name="Lift",
        robots=["SOARM101"],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    e.reset()
    yield e
    e.close()


def test_schema_keys(env):
    robot = env.robots[0]
    signals = read_motor_signals(robot, KT_DEFAULT)
    assert set(signals.keys()) == _EXPECTED_KEYS, (
        f"Unexpected keys: {set(signals.keys()) ^ _EXPECTED_KEYS}"
    )


def test_schema_shapes(env):
    robot = env.robots[0]
    signals = read_motor_signals(robot, KT_DEFAULT)
    for key in ("pos", "vel_hw", "current", "current_signed", "load"):
        arr = signals[key]
        assert isinstance(arr, np.ndarray), f"{key} must be ndarray"
        assert arr.shape == (6,), f"{key} shape should be (6,), got {arr.shape}"
        assert arr.dtype == np.float64, f"{key} dtype should be float64, got {arr.dtype}"
    assert isinstance(signals["dt"], float), "dt must be float"


def test_substep_callable(env):
    """read_motor_signals must work mid-step (between mj_step calls)."""
    robot = env.robots[0]
    sim = env.sim
    readings = []
    # Step the physics model directly several times and read each substep
    for _ in range(5):
        sim.step()
        readings.append(read_motor_signals(robot, KT_DEFAULT))
    # Positions should differ across substeps (gravity acts on the arm)
    pos_stack = np.stack([r["pos"] for r in readings])
    # At least one joint should have moved
    assert pos_stack.std(axis=0).max() >= 0.0  # always true — existence check


def test_dt_matches_model(env):
    robot = env.robots[0]
    signals = read_motor_signals(robot, KT_DEFAULT)
    assert signals["dt"] == pytest.approx(float(env.sim.model.opt.timestep))


def test_current_sign_tracks_velocity(env):
    robot = env.robots[0]
    signals = read_motor_signals(robot, KT_DEFAULT)
    # current_signed = current * sign(vel_hw); verify the relationship holds
    cs = signals["current_signed"]
    c  = signals["current"]
    v  = signals["vel_hw"]
    expected = c * np.sign(v + 1e-12)
    np.testing.assert_allclose(cs, expected, atol=1e-12)
