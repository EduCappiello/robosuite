import numpy as np

from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift
from robosuite.utils.observables import Observable, sensor


class SOARM101PnPCup(SOARM101Lift):
    """
    Pick-and-place task for SOARM101: move the cup from the table into the
    coffee machine cup bay.

    This task intentionally reuses SOARM101Lift's scene construction so the
    tuned coffee machine, cup, gripper camera, friction, and collision setup stay
    in one place. The task-specific part is the coffee-bay target volume,
    success condition, and reward.
    """

    def __init__(
        self,
        *args,
        coffee_target_half_size=(0.035, 0.040, 0.015),
        cup_start_offset=(-0.11, 0.15, 0.0),
        cup_start_yaw_deg=0.0,
        success_hold_steps=3,
        cup_max_tilt_deg=45.0,
        cup_failure_tilt_deg=80.0,
        reward_shaping=True,
        **kwargs,
    ):
        self.coffee_target_half_size = np.array(coffee_target_half_size, dtype=float)
        self.cup_start_offset = np.array(cup_start_offset, dtype=float)
        self.cup_start_yaw_deg = float(cup_start_yaw_deg)
        self.success_hold_steps = int(success_hold_steps)
        self.cup_max_tilt_deg = float(cup_max_tilt_deg)
        self.cup_failure_tilt_deg = float(cup_failure_tilt_deg)
        self._success_counter = 0
        kwargs.setdefault("coffee_machine_offset", (0.17, 0.0, 0.0))
        super().__init__(*args, reward_shaping=reward_shaping, **kwargs)

    def _load_model(self):
        super()._load_model()


    def _cup_bay_center_and_bounds(self):
        """Return the physically usable cup-center target on the machine tray."""
        cup_half_height = float(self.cup_size[1])
        center = np.array(self.coffee_target_center, dtype=float)
        center[2] = self.coffee_tray_top_z + cup_half_height + 0.002

        half_size = np.array(self.coffee_target_half_size, dtype=float)
        low = center - half_size
        high = center + half_size

        # A successful cup must have its full 6 cm footprint supported by the tray.
        low[0] = max(low[0], float(self.coffee_tray_center_x_bounds[0]))
        high[0] = min(high[0], float(self.coffee_tray_center_x_bounds[1]))
        if np.any(low >= high):
            raise ValueError(f"Coffee target has no usable volume: low={low}, high={high}")

        center = 0.5 * (low + high)
        return center, low, high

    def _cup_pos(self):
        return np.array(self.sim.data.body_xpos[self.cube_body_id], dtype=float)

    def _cup_in_target(self):
        cup_pos = self._cup_pos()
        center, low, high = self._cup_bay_center_and_bounds()
        center_tolerance = self.coffee_pad_radius - float(self.cup_size[0])
        centered_on_pad = np.linalg.norm(cup_pos[:2] - center[:2]) <= center_tolerance
        at_target_height = low[2] <= cup_pos[2] <= high[2]
        return bool(centered_on_pad and at_target_height)

    def _cup_is_upright(self):
        cup_rotation = np.array(self.sim.data.body_xmat[self.cube_body_id], dtype=float).reshape(3, 3)
        cup_up_world = cup_rotation[:, 2]
        return bool(cup_up_world[2] >= np.cos(np.deg2rad(self.cup_max_tilt_deg)))

    def _cup_exceeds_failure_tilt(self):
        cup_rotation = np.array(self.sim.data.body_xmat[self.cube_body_id], dtype=float).reshape(3, 3)
        cup_up_world = cup_rotation[:, 2]
        return bool(cup_up_world[2] < np.cos(np.deg2rad(self.cup_failure_tilt_deg)))
    @property
    def task_success(self):
        return self._success_counter >= self.success_hold_steps

    @property
    def task_failed(self):
        """True once the cup has tipped beyond the allowed upright angle."""
        return self._cup_exceeds_failure_tilt()

    def _update_success_target_visual(self, success):
        """The physical cup pad remains white before and after success."""
        del success

    def _check_success(self):
        """True once the cup center has stayed inside the coffee bay briefly."""
        if self._cup_in_target() and self._cup_is_upright():
            self._success_counter += 1
        else:
            self._success_counter = 0
        success = self.task_success
        self._update_success_target_visual(success)
        return success

    def reward(self, action=None):
        """Sparse success reward plus optional dense reaching / placing terms."""
        if self._check_success():
            return 1.0 if self.reward_scale is None else float(self.reward_scale)

        reward = 0.0
        if self.reward_shaping:
            cup_pos = self._cup_pos()
            target_pos, _, _ = self._cup_bay_center_and_bounds()

            gripper_to_cup = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )
            cup_to_target = float(np.linalg.norm(cup_pos - target_pos))

            reach_reward = 0.35 * (1.0 - np.tanh(8.0 * gripper_to_cup))
            place_reward = 0.55 * (1.0 - np.tanh(6.0 * cup_to_target))
            grasp_reward = 0.10 if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube) else 0.0
            reward = reach_reward + place_reward + grasp_reward

        if self.reward_scale is not None:
            reward *= float(self.reward_scale)
        return float(reward)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def coffee_target_pos(obs_cache):
                center, _, _ = self._cup_bay_center_and_bounds()
                return center

            @sensor(modality=modality)
            def cup_to_coffee_target(obs_cache):
                center, _, _ = self._cup_bay_center_and_bounds()
                return center - self._cup_pos()

            @sensor(modality=modality)
            def cup_in_coffee_target(obs_cache):
                return np.array([float(self._cup_in_target())])

            @sensor(modality=modality)
            def cup_upright(obs_cache):
                return np.array([float(self._cup_is_upright())])

            for s in (coffee_target_pos, cup_to_coffee_target, cup_in_coffee_target, cup_upright):
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        self._success_counter = 0
        super()._reset_internal()
        self._reset_cup_to_table_start()
        self._update_success_target_visual(False)

    def _reset_cup_to_table_start(self):
        """Place the cup on the table, outside the coffee machine, at reset."""
        _, cup_half_height = [float(v) for v in self.cup_size]
        pos = np.array(self.table_offset, dtype=float) + self.cup_start_offset
        pos[2] = float(self.table_offset[2] + cup_half_height + 0.005)

        yaw = np.deg2rad(self.cup_start_yaw_deg)
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, quat]))