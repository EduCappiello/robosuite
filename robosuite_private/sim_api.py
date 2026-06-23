"""
SOARM101Sim — thin facade over a robosuite SOARM101Lift environment.

Exposes reset / step / substep-hook interfaces so lerobot-private can:
  1. Drive the sim with absolute joint targets.
  2. Sample read_motor_signals and get_ground_truth_dynamics per physics
     substep (not only per control step) by registering a substep_hook.
  3. Read camera frames without touching robosuite internals directly.

lerobot-private imports this module directly (same conda environment).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SOARM101Sim:
    """
    Thin facade over robosuite's SOARM101Lift environment.

    Args:
        env_kwargs: keyword arguments forwarded to suite.make().
                    "env_name" defaults to "SOARM101Lift".
                    "robots"   defaults to ["SOARM101"].
    """

    def __init__(self, env_kwargs: dict | None = None) -> None:
        import robosuite as suite
        from robosuite.utils.binding_utils import MjSim

        kwargs: dict = {
            "env_name":                "SOARM101Lift",
            "robots":                  ["SOARM101"],
            "has_renderer":            False,
            "has_offscreen_renderer":  False,
            "ignore_done":             True,
            "use_camera_obs":          False,
            "control_freq":            20,
        }
        if env_kwargs:
            kwargs.update(env_kwargs)

        self._env    = suite.make(**kwargs)
        self._robot  = None            # set after first reset()
        self._hooks: list[Callable] = []

        # number of physics substeps per control step
        self._n_substeps: int = self._env.control_timestep // self._env.model_timestep  # type: ignore[operator]

    # ------------------------------------------------------------------ public

    @property
    def env(self):
        """Direct access to the underlying robosuite environment."""
        return self._env

    @property
    def robot(self):
        """The primary robot instance (None before first reset)."""
        return self._robot

    def substep_hook(self, fn: Callable) -> None:
        """
        Register a callable invoked after every physics substep.

        The callable receives (robot, sim_data):
            robot    — robosuite robot instance
            sim_data — dict from the current sim state with keys:
                       "qpos", "qvel", "qacc", "qfrc_actuator"

        Hooks are cleared on each reset().  Register after reset() if you want
        them to persist for the episode.
        """
        self._hooks.append(fn)

    def reset(self) -> dict:
        """
        Reset the environment and return the initial observation.

        Clears any registered substep hooks and re-sets self.robot.
        """
        obs = self._env.reset()
        self._robot = self._env.robots[0]
        self._hooks = []
        return obs

    def step(self, goal_positions: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Step the environment with absolute joint position targets.

        goal_positions: action array matching env.action_dim (controller-dependent).
                        For JOINT_POSITION: (6,) = [5 arm rad + 1 gripper].
                        For OSC_POSITION:   (4,) = [3 pos delta + 1 gripper].

        Substep hooks are called after each of the n_substeps physics steps.

        Returns:
            (obs, reward, done, info) matching robosuite's env.step() output.
        """
        action = np.asarray(goal_positions, dtype=np.float64)
        return self._env.step(action)

    def render(self) -> None:
        """Render the on-screen MuJoCo viewer.

        Requires ``has_renderer=True`` at construction (the env builds the GLFW window
        lazily on the first render call). No-op-safe to call every control step; used by
        interactive drivers (e.g. lerobot teleoperation) to watch the arm move.
        """
        self._env.render()

    def _step_with_hooks(self, goal_positions: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Step with per-substep hook firing.

        Bypasses robosuite's step() to give hooks visibility at the MuJoCo
        timestep rate.  Only use when hooks are registered; otherwise use
        the standard step() for efficiency.
        """
        if not self._hooks:
            return self.step(goal_positions)

        # Drive the controller for this control step's n_substeps
        # robosuite sets ctrl targets once per control step; we replicate that.
        env  = self._env
        robot = self._robot
        sim  = env.sim

        # Let the composite controller compute actuator signals
        env._pre_action(np.asarray(goal_positions))

        for _ in range(int(env.control_timestep / env.model_timestep)):
            sim.step()
            if self._hooks:
                snap = {
                    "qpos":          np.array(sim.data.qpos),
                    "qvel":          np.array(sim.data.qvel),
                    "qacc":          np.array(sim.data.qacc),
                    "qfrc_actuator": np.array(sim.data.qfrc_actuator),
                }
                for fn in self._hooks:
                    fn(robot, snap)

        obs    = env._get_observations()
        reward = env.reward(np.asarray(goal_positions))
        done   = env._check_success() or env.done
        info   = {}
        return obs, reward, done, info

    def get_camera_frames(
        self,
        camera_names: list[str] | None = None,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, np.ndarray]:
        """Render camera frames from the current simulation state."""
        from robosuite_private.rendering.cameras import get_camera_frames
        return get_camera_frames(self._env, camera_names, width, height)

    def close(self) -> None:
        self._env.close()
        self._robot = None

    # ------------------------------------------------------------------ context manager

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
