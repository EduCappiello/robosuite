"""
Framework-agnostic test: importing robosuite_private must not pull in any
LeRobot or estimation-core modules.
"""

import sys
import importlib
import pytest

_FORBIDDEN_PREFIXES = (
    "lerobot",
    "lerobot_private",
    "estimation_core",
)


def test_no_lerobot_on_import():
    # Reload in a clean namespace snapshot so we don't get false positives
    # from other tests that may have imported LeRobot.
    # We check the modules that robosuite_private itself causes to be loaded.
    before = set(sys.modules.keys())

    import robosuite_private  # noqa: F401
    import robosuite_private.robot_spec  # noqa: F401
    import robosuite_private.motor_signals  # noqa: F401
    import robosuite_private.dynamics_gt  # noqa: F401
    import robosuite_private.noise  # noqa: F401
    import robosuite_private.rendering  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before

    leaky = [m for m in new_modules if any(m.startswith(p) for p in _FORBIDDEN_PREFIXES)]
    assert not leaky, (
        f"robosuite_private imported LeRobot/estimation modules: {leaky}"
    )
