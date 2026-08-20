import numpy as np

from robosuite.environments.manipulation.soarm101_lift import (
    XLEROBOT_ARM_DECK_Z,
    XLEROBOT_ARM_FORWARD_X,
    SOARM101Lift,
    robots_bring_their_own_base,
)
from robosuite.utils.observables import Observable, sensor

class cupPnP_task1(SOARM101Lift):
    """
    Independent cup pick-and-place task with the task1 table layout.

    Its sensors, reward, reset behavior, and success / failure conditions match
    SOARM101PnPCup. Only the main-table height, support table, robot base, cup,
    and coffee-machine positions differ.
    """

    def __init__(
        self,
        *args,
        coffee_target_half_size=(0.035, 0.040, 0.015),
        cup_start_offset=(-0.29, 0.22, 0.0),
        cup_start_yaw_deg=0.0,
        cup_position_randomization_m=0.010,
        cup_yaw_randomization_deg=7.5,
        cup_color_randomization=0.03,
        success_hold_steps=3,
        cup_max_tilt_deg=45.0,
        cup_failure_tilt_deg=80.0,
        gripper_clearance_m=0.08,
        target_center_margin_m=0.005,
        cart_full_size=(0.35, 0.45, 0.04),
        cart_top_height=0.77,
        main_table_top_height=0.82,
        cart_gap=0.01,
        robot_on_cart_offset=(-0.07, 0.10, 0.0),
        coffee_machine_offset=(-0.1965, 0.0, 0.0),
        head_camera_robot_offset=(-0.402, -0.250, 0.46065),
        head_start_pan_deg=8.0,
        head_start_tilt_deg=32.0,
        reward_shaping=True,
        **kwargs,
    ):
        self.coffee_target_half_size = np.array(coffee_target_half_size, dtype=float)
        self.cup_start_offset = np.array(cup_start_offset, dtype=float)
        self.cup_start_yaw_deg = float(cup_start_yaw_deg)
        self.cup_position_randomization_m = float(cup_position_randomization_m)
        self.cup_yaw_randomization_deg = float(cup_yaw_randomization_deg)
        self.cup_color_randomization = float(cup_color_randomization)
        self.success_hold_steps = int(success_hold_steps)
        self.cup_max_tilt_deg = float(cup_max_tilt_deg)
        self.cup_failure_tilt_deg = float(cup_failure_tilt_deg)
        self.gripper_clearance_m = float(gripper_clearance_m)
        self.target_center_margin_m = float(target_center_margin_m)
        self._success_counter = 0
        self.head_start_pan_deg = float(head_start_pan_deg)
        self.head_start_tilt_deg = float(head_start_tilt_deg)

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
        kwargs.setdefault("robot_support_table_full_size", cart_full_size)
        kwargs.setdefault("robot_support_table_offset", cart_top)
        kwargs.setdefault("robot_support_robot_offset", robot_on_cart_offset)
        self.robot_base_pos = (
            np.array(kwargs["robot_support_table_offset"], dtype=float)
            + np.array(kwargs["robot_support_robot_offset"], dtype=float)
        )
        if robots_bring_their_own_base(kwargs.get("robots")):
            # robot_base_pos only drives the head-camera extrinsic, which is defined
            # relative to the ARM mount. For a self-supporting robot the base frame is
            # on the floor, so point this at the midpoint of the two arm bases instead
            # — that keeps the cameras framed as they are for the desk arm.
            self.robot_base_pos = np.array(
                [cart_center_x + XLEROBOT_ARM_FORWARD_X, 0.0, XLEROBOT_ARM_DECK_Z],
                dtype=float,
            )
        super().__init__(*args, reward_shaping=reward_shaping, **kwargs)

    def _load_model(self):
        # TableArena defines table_offset Z as the tabletop surface.
        self.table_offset = np.array([0.0, 0.0, self.main_table_top_height], dtype=float)
        super()._load_model()

        # Keep the physical head cameras at the same robot-relative extrinsic
        # as SOARM101PnPCup. Their intrinsics and orientation come unchanged
        # from SOARM101Lift; only the world-space position follows this robot.
        head_camera_pos = self.robot_base_pos + self.head_camera_robot_offset
        for camera_name in ("top", "rear"):
            camera = self.model.worldbody.find(f".//camera[@name='{camera_name}']")
            if camera is None:
                raise ValueError(f"Missing expected camera: {camera_name}")
            camera.set("pos", " ".join(str(float(v)) for v in head_camera_pos))


    def _cup_bay_center_and_bounds(self):
        """Return the physically usable cup-center target on the machine tray."""
        cup_half_height = float(self.cup_size[1])
        center = np.array(self.coffee_target_center, dtype=float)
        center[2] = self.coffee_tray_top_z + cup_half_height + 0.002

        half_size = np.array(self.coffee_target_half_size, dtype=float)
        low = center - half_size
        high = center + half_size

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
        center_tolerance = (
            self.coffee_pad_radius
            - float(self.cup_size[0])
            + self.target_center_margin_m
        )
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

    def _cup_is_released(self):
        return not self._task_gripper_contacts_object(self.cube)

    def _gripper_is_clear(self):
        distance = np.linalg.norm(self._task_gripper_tip_pos() - self._cup_pos())
        return bool(distance >= self.gripper_clearance_m)

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
        """True once the upright cup is placed and the gripper has withdrawn."""
        if (
            self._cup_in_target()
            and self._cup_is_upright()
            and self._cup_is_released()
            and self._gripper_is_clear()
        ):
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

            for observable_sensor in (
                coffee_target_pos,
                cup_to_coffee_target,
                cup_in_coffee_target,
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
        if self.robot_is_self_supporting:
            for joint_name, angle_deg in (
                ("robot0_head_pan", self.head_start_pan_deg),
                ("robot0_head_tilt", self.head_start_tilt_deg),
            ):
                self.sim.data.set_joint_qpos(joint_name, np.deg2rad(angle_deg))
                self.sim.data.set_joint_qvel(joint_name, 0.0)
        self._reset_cup_to_table_start()
        self._randomize_cup_visual_color(self.cup_color_randomization)
        self.sim.forward()
        self._update_success_target_visual(False)

    def _reset_cup_to_table_start(self):
        """Place the cup on the table, outside the coffee machine, at reset."""
        _, cup_half_height = [float(v) for v in self.cup_size]
        pos = np.array(self.table_offset, dtype=float) + self.cup_start_offset
        pos[:2] += self._sample_cup_xy_jitter(self.cup_position_randomization_m)
        pos[2] = float(self.table_offset[2] + cup_half_height + 0.005)

        yaw_deg = self.cup_start_yaw_deg + self._sample_cup_yaw_jitter_deg(
            self.cup_yaw_randomization_deg
        )
        yaw = np.deg2rad(yaw_deg)
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, quat]))
        self.sim.data.set_joint_qvel(self.cube.joints[0], np.zeros(6))
