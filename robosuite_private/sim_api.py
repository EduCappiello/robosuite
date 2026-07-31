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
        self._viewer_camera_toggle = None
        self._viewer_camera_toggle_attached = False
        self._stop_requested = False

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

    @property
    def stop_requested(self) -> bool:
        """Whether the interactive viewer requested a clean driver shutdown."""
        return self._stop_requested

    def request_stop(self) -> None:
        """Request that the outer teleoperation loop stop at its next safe point."""
        self._stop_requested = True

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
        old_viewer = getattr(getattr(self._env, "viewer", None), "viewer", None)
        obs = self._env.reset()
        new_viewer = getattr(getattr(self._env, "viewer", None), "viewer", None)
        if old_viewer is not new_viewer:
            self._viewer_camera_toggle_attached = False
        self._robot = self._env.robots[0]
        self._hooks = []
        self._stop_requested = False
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
        self._attach_viewer_camera_toggle()

    def configure_viewer_camera_toggle(
        self,
        *,
        camera_name: str | None = "robot0_gripper_cam",
        key: str | None = "C",
        start_locked: bool = False,
    ) -> None:
        """Configure viewer camera keys and the Q/Esc clean-stop hotkey.

        Keys in the MuJoCo viewer:
          C/G: lock to the configured camera (when enabled)
          F: return to free mouse camera (when enabled)
          Q/Esc: request a clean shutdown of the outer driver loop
        """
        self._viewer_camera_toggle = (
            {
                "camera_name": camera_name,
                "key": key.upper()[0],
                "locked": bool(start_locked),
                "start_locked": bool(start_locked),
            }
            if camera_name and key
            else None
        )
        self._viewer_camera_toggle_attached = False
        self._attach_viewer_camera_toggle()

    def _attach_viewer_camera_toggle(self) -> None:
        cfg = self._viewer_camera_toggle
        if self._viewer_camera_toggle_attached:
            return
        if not getattr(self._env, "has_renderer", False):
            return

        viewer = getattr(self._env, "viewer", None)
        if viewer is None or not hasattr(viewer, "add_keypress_callback"):
            return

        def set_viewer_camera(locked: bool) -> None:
            if cfg is None:
                return
            camera_name = cfg["camera_name"]
            camera_id = self._env.sim.model.camera_name2id(camera_name) if locked else -1
            if locked and camera_id < 0:
                print(f"[robosuite] Viewer camera {camera_name!r} not found; staying in free camera.")
                cfg["locked"] = False
                viewer.set_camera(-1)
                return
            cfg["locked"] = locked
            viewer.set_camera(camera_id)
            mode = camera_name if locked else "free mouse camera"
            print(f"[robosuite] Viewer camera: {mode}")

        def on_keypress(keycode: int) -> None:
            # GLFW_KEY_ESCAPE is 256. Q is also checked by character so this
            # remains independent of which window owns the console focus.
            if keycode == 256:
                self.request_stop()
                print("[robosuite] Viewer stop requested; shutting down cleanly...")
                return
            try:
                pressed = chr(keycode).upper()
            except (TypeError, ValueError):
                return
            if pressed == "Q":
                self.request_stop()
                print("[robosuite] Viewer stop requested; shutting down cleanly...")
            elif cfg is not None and (pressed == "G" or pressed == cfg["key"]):
                set_viewer_camera(True)
            elif cfg is not None and pressed == "F":
                set_viewer_camera(False)

        viewer.add_keypress_callback(on_keypress)
        self._viewer_camera_toggle_attached = True
        if cfg is None:
            print("[robosuite] Viewer hotkeys: Q/Esc=stop")
        else:
            print(
                f"[robosuite] Viewer hotkeys: {cfg['key']}/G={cfg['camera_name']}, "
                "F=free mouse camera, Q/Esc=stop"
            )
        if cfg is not None and cfg["start_locked"]:
            set_viewer_camera(True)
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

    def get_camera_rgbd_frames(
        self,
        camera_names: list[str] | None = None,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Render RGB and metric uint16 depth from the current simulation state."""
        from robosuite_private.rendering.cameras import get_camera_rgbd_frames

        return get_camera_rgbd_frames(self._env, camera_names, width, height)

    def get_task_state(self) -> dict[str, float]:
        """Return simulation-only task labels for dataset recording.

        PnP environments expose the manipulated cup through env.cube for
        compatibility with the original Lift task. The free-joint qpos is in
        world coordinates and uses MuJoCo's [x, y, z, qw, qx, qy, qz] order.
        """
        state = {
            "cup_pos_x": 0.0,
            "cup_pos_y": 0.0,
            "cup_pos_z": 0.0,
            "cup_quat_w": 1.0,
            "cup_quat_x": 0.0,
            "cup_quat_y": 0.0,
            "cup_quat_z": 0.0,
            "success": 0.0,
            "failure": 0.0,
        }

        cup = getattr(self._env, "cube", None)
        if cup is not None and getattr(cup, "joints", None):
            qpos = np.asarray(self._env.sim.data.get_joint_qpos(cup.joints[0]), dtype=float)
            if qpos.shape[0] >= 7:
                state.update(
                    {
                        "cup_pos_x": float(qpos[0]),
                        "cup_pos_y": float(qpos[1]),
                        "cup_pos_z": float(qpos[2]),
                        "cup_quat_w": float(qpos[3]),
                        "cup_quat_x": float(qpos[4]),
                        "cup_quat_y": float(qpos[5]),
                        "cup_quat_z": float(qpos[6]),
                    }
                )

        success = getattr(self._env, "task_success", False)
        failure = getattr(self._env, "task_failed", False)
        state["success"] = float(success) if not callable(success) else 0.0
        state["failure"] = float(failure) if not callable(failure) else 0.0
        return state

    def close(self) -> None:
        self._env.close()
        self._robot = None

    # ------------------------------------------------------------------ context manager

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class XLeRobotSim(SOARM101Sim):
    """
    SOARM101Sim facade with XLeRobot defaults: the 17-DoF dual-arm mobile robot
    in a plain Lift scene (coffee tasks are re-parented onto XLeRobot in M4).

    Action layout with default_xlerobot.json (17,), composite order:
        [right arm 5, left arm 5, head 2, base vel 3, right grip 1, left grip 1]
    (build actions with robot.create_action_vector to stay layout-proof).

    Per-arm ground truth / motor signals: pass arm="right"/"left" to
    get_ground_truth_dynamics(...) and read_motor_signals(...).
    """

    def __init__(self, env_kwargs: dict | None = None) -> None:
        kwargs: dict = {
            "env_name": "Lift",
            "robots": ["XLeRobot"],
            # Deterministic reset: identity tests and estimator comparisons
            # assume the exact in-range init pose (see test_external_gt).
            "initialization_noise": None,
        }
        if env_kwargs:
            kwargs.update(env_kwargs)
        super().__init__(kwargs)
