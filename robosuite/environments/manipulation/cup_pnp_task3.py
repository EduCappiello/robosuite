import numpy as np

from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift
from robosuite.utils.observables import Observable, sensor


class cupPnP_task3(SOARM101Lift):
    """Move a cup from the coffee machine onto the robot's support table."""

    def __init__(
        self,
        *args,
        coffee_target_half_size=(0.035, 0.040, 0.015),
        cup_start_yaw_deg=0.0,
        cart_target_offset=(0.11, -0.16, 0.0),
        success_hold_steps=3,
        cup_max_tilt_deg=45.0,
        cup_failure_tilt_deg=80.0,
        cart_full_size=(0.35, 0.45, 0.04),
        cart_top_height=0.77,
        main_table_top_height=0.865,
        cart_gap=0.01,
        robot_on_cart_offset=(-0.07, 0.10, 0.0),
        coffee_machine_offset=(-0.16, 0.0, 0.0),
        head_camera_robot_offset=(-0.402, -0.250, 0.46065),
        reward_shaping=True,
        **kwargs,
    ):
        self.coffee_target_half_size = np.array(coffee_target_half_size, dtype=float)
        self.cup_start_yaw_deg = float(cup_start_yaw_deg)
        self.cart_target_offset = np.array(cart_target_offset, dtype=float)
        self.success_hold_steps = int(success_hold_steps)
        self.cup_max_tilt_deg = float(cup_max_tilt_deg)
        self.cup_failure_tilt_deg = float(cup_failure_tilt_deg)
        self._success_counter = 0

        table_full_size = np.array(kwargs.get("table_full_size", (0.8, 0.8, 0.05)), dtype=float)
        cart_full_size = np.array(cart_full_size, dtype=float)
        cart_center_x = -(table_full_size[0] + cart_full_size[0]) / 2.0 - float(cart_gap)
        cart_top = np.array([cart_center_x, 0.0, float(cart_top_height)], dtype=float)

        self.cart_full_size = cart_full_size
        self.cart_top = cart_top
        self.main_table_top_height = float(main_table_top_height)
        self.cart_gap = float(cart_gap)
        self.head_camera_robot_offset = np.array(head_camera_robot_offset, dtype=float)

        kwargs.setdefault("coffee_machine_offset", coffee_machine_offset)
        kwargs.setdefault("cup_model_path", "objects/coffee_cup_3_water/model.xml")
        kwargs.setdefault("robot_support_table_full_size", cart_full_size)
        kwargs.setdefault("robot_support_table_offset", cart_top)
        kwargs.setdefault("robot_support_robot_offset", robot_on_cart_offset)
        self.robot_base_pos = (
            np.array(kwargs["robot_support_table_offset"], dtype=float)
            + np.array(kwargs["robot_support_robot_offset"], dtype=float)
        )
        super().__init__(*args, reward_shaping=reward_shaping, **kwargs)

    def _load_model(self):
        self.table_offset = np.array([0.0, 0.0, self.main_table_top_height], dtype=float)
        super()._load_model()

        head_camera_pos = self.robot_base_pos + self.head_camera_robot_offset
        for camera_name in ("top", "rear"):
            camera = self.model.worldbody.find(f".//camera[@name='{camera_name}']")
            if camera is None:
                raise ValueError(f"Missing expected camera: {camera_name}")
            camera.set("pos", " ".join(str(float(v)) for v in head_camera_pos))

        # SOARM101Lift already creates a blue X. Move it from the main table to
        # the support table's upper-right corner (forward and robot-right).
        marker_pos = self._cart_target_pos()
        marker_pos[2] = self.cart_top[2] + 0.0015
        marker_pos_text = " ".join(str(float(v)) for v in marker_pos)
        for geom_name in ("target_x_a", "target_x_b"):
            marker = self.model.worldbody.find(f".//geom[@name='{geom_name}']")
            if marker is None:
                raise ValueError(f"Missing expected target marker: {geom_name}")
            marker.set("pos", marker_pos_text)

    def _cup_bay_center_and_bounds(self):
        """Return the physically usable cup-center region on the machine tray."""
        cup_half_height = float(self.cup_size[1])
        center = np.array(self.coffee_target_center, dtype=float)
        center[2] = self.coffee_tray_top_z + cup_half_height + 0.002

        half_size = np.array(self.coffee_target_half_size, dtype=float)
        low = center - half_size
        high = center + half_size
        low[0] = max(low[0], float(self.coffee_tray_center_x_bounds[0]))
        high[0] = min(high[0], float(self.coffee_tray_center_x_bounds[1]))
        if np.any(low >= high):
            raise ValueError(f"Coffee cup start has no usable volume: low={low}, high={high}")

        return 0.5 * (low + high), low, high

    def _cart_target_pos(self):
        """Return the visual target center, expressed in world coordinates."""
        target = self.cart_top + self.cart_target_offset
        target[2] = self.cart_top[2] + float(self.cup_size[1]) + 0.002
        return target

    def _cart_cup_center_xy_bounds(self):
        """Return the complete support-table footprint in world XY."""
        half_xy = self.cart_full_size[:2] / 2.0
        low = self.cart_top[:2] - half_xy
        high = self.cart_top[:2] + half_xy
        return low, high

    def _cup_pos(self):
        return np.array(self.sim.data.body_xpos[self.cube_body_id], dtype=float)

    def _cup_on_cart(self):
        cup_pos = self._cup_pos()
        low, high = self._cart_cup_center_xy_bounds()
        inside_footprint = bool(np.all(cup_pos[:2] >= low) and np.all(cup_pos[:2] <= high))

        cup_bottom_z = cup_pos[2] - float(self.cup_size[1])
        surface_delta = cup_bottom_z - float(self.cart_top[2])
        near_tabletop = -0.015 <= surface_delta <= 0.020
        return inside_footprint and near_tabletop

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
        return self._cup_exceeds_failure_tilt()

    def _update_target_visual(self, success):
        color = (
            np.array([0.15, 1.0, 0.15, 0.9])
            if success
            else np.array([0.05, 0.2, 0.85, 1.0])
        )
        for geom_name in ("target_x_a", "target_x_b"):
            geom_id = self.sim.model.geom_name2id(geom_name)
            self.sim.model.geom_rgba[geom_id] = color

    def _check_success(self):
        if self._cup_on_cart() and self._cup_is_upright():
            self._success_counter += 1
        else:
            self._success_counter = 0
        success = self.task_success
        self._update_target_visual(success)
        return success

    def reward(self, action=None):
        if self._check_success():
            return 1.0 if self.reward_scale is None else float(self.reward_scale)

        reward = 0.0
        if self.reward_shaping:
            cup_pos = self._cup_pos()
            target_pos = self._cart_target_pos()
            gripper_to_cup = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )
            cup_to_target = float(np.linalg.norm(cup_pos - target_pos))
            reach_reward = 0.35 * (1.0 - np.tanh(8.0 * gripper_to_cup))
            place_reward = 0.55 * (1.0 - np.tanh(6.0 * cup_to_target))
            grasp_reward = (
                0.10
                if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
                else 0.0
            )
            reward = reach_reward + place_reward + grasp_reward

        if self.reward_scale is not None:
            reward *= float(self.reward_scale)
        return float(reward)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cart_target_pos(obs_cache):
                return self._cart_target_pos()

            @sensor(modality=modality)
            def cup_to_cart_target(obs_cache):
                return self._cart_target_pos() - self._cup_pos()

            @sensor(modality=modality)
            def cup_on_cart(obs_cache):
                return np.array([float(self._cup_on_cart())])

            @sensor(modality=modality)
            def cup_upright(obs_cache):
                return np.array([float(self._cup_is_upright())])

            for observable_sensor in (
                cart_target_pos,
                cup_to_cart_target,
                cup_on_cart,
                cup_upright,
            ):
                observables[observable_sensor.__name__] = Observable(
                    name=observable_sensor.__name__,
                    sensor=observable_sensor,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        self._success_counter = 0
        super()._reset_internal()
        self._reset_cup_to_coffee_start()

    def _reset_cup_to_coffee_start(self):
        """Place the cup upright in the coffee-machine bay at reset."""
        pos, _, _ = self._cup_bay_center_and_bounds()
        yaw = np.deg2rad(self.cup_start_yaw_deg)
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, quat]))
