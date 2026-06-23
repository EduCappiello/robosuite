"""
Substep test: verify that substep hooks fire per physics step and that dt
reported by read_motor_signals matches model.opt.timestep.
"""

import numpy as np
import pytest

from robosuite_private.sim_api import SOARM101Sim
from robosuite_private.motor_signals import read_motor_signals
from robosuite_private.robot_spec import KT_DEFAULT


def test_substep_hook_fires_per_physics_step():
    """Hook must fire n_substeps times per control step."""
    sim = SOARM101Sim()
    sim.reset()

    n_substeps = int(sim.env.control_timestep / sim.env.model_timestep)

    call_count = []

    def counter(robot, snap):
        call_count.append(1)

    sim.substep_hook(counter)
    action = np.zeros(sim.env.action_dim)
    sim._step_with_hooks(action)

    assert len(call_count) == n_substeps, (
        f"Hook fired {len(call_count)} times, expected {n_substeps}"
    )
    sim.close()


def test_hook_receives_valid_snap():
    """Snapshot dict passed to hook must contain the required keys."""
    sim = SOARM101Sim()
    sim.reset()

    snaps = []

    def capture(robot, snap):
        snaps.append(snap)

    sim.substep_hook(capture)
    sim._step_with_hooks(np.zeros(sim.env.action_dim))

    assert snaps, "No snapshots captured"
    required = {"qpos", "qvel", "qacc", "qfrc_actuator"}
    assert required.issubset(snaps[0].keys())
    sim.close()


def test_dt_matches_model_timestep():
    sim = SOARM101Sim()
    sim.reset()
    robot = sim.robot
    signals = read_motor_signals(robot, KT_DEFAULT)
    assert signals["dt"] == pytest.approx(float(sim.env.sim.model.opt.timestep))
    sim.close()
