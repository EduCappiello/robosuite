import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift
from robosuite.utils.observables import Observable, sensor

# Footprint (x, y) of the XLeRobot RASKOG cart and the height of its arm deck.
# Used to park a self-supporting robot exactly where the support table would sit.
XLEROBOT_CART_FOOTPRINT = (0.392, 0.467)
XLEROBOT_ARM_DECK_Z = 0.8215
# Arm bases sit on the top plate's mounting pads, BEHIND the cart centre (negative x).
# Keep in sync with XLEROBOT_FRAME_TREE.
XLEROBOT_ARM_FORWARD_X = -0.0911


def _robots_bring_their_own_base(robots):
    """True if any requested robot is mobile, i.e. it stands on the floor on its own
    chassis and must NOT be perched on the fixed support table."""
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.mobile_robot import MobileRobot

    if robots is None:
        return False
    names = [robots] if isinstance(robots, str) else list(robots)
    for name in names:
        cls = ROBOT_CLASS_MAPPING.get(name)
        if cls is not None and issubclass(cls, MobileRobot):
            return True
    return False


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
        cup_start_offset=(-0.37, 0.22, 0.0),
        cup_start_yaw_deg=0.0,
        success_hold_steps=3,
        cup_max_tilt_deg=45.0,
        cup_failure_tilt_deg=80.0,
        cart_full_size=(0.35, 0.45, 0.04),
        cart_top_height=0.77,
        main_table_top_height=0.865,
        cart_gap=0.01,
        robot_on_cart_offset=(-0.07, 0.10, 0.0),
        coffee_machine_offset=(-0.19, 0.0, 0.0),
        head_camera_robot_offset=(-0.402, -0.250, 0.46065),
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

        table_full_size = np.array(kwargs.get("table_full_size", (0.8, 0.8, 0.05)), dtype=float)

        cart_full_size = np.array(cart_full_size, dtype=float)
        cart_center_x = -(table_full_size[0] + cart_full_size[0]) / 2.0 - float(cart_gap)
        cart_top = np.array([cart_center_x, 0.0, float(cart_top_height)], dtype=float)

        # A mobile robot (XLeRobot) arrives on its own RASKOG cart, so it REPLACES the
        # support table rather than standing on it: the table is not built at all and
        # the robot is parked on the floor beside the main table.
        #
        # It is parked so its ARM BASES land where the desk arm's base was, NOT so its
        # chassis lands where the support table was. The arms sit 135 mm forward of the
        # cart centre, so matching chassis-to-chassis would push both arms 18 cm closer
        # to the table than the task was designed for — they spawn inside the coffee
        # machine. Matching arm-base to arm-base preserves the reach geometry.
        self.robot_is_self_supporting = _robots_bring_their_own_base(kwargs.get("robots"))
        if self.robot_is_self_supporting:
            desk_arm_base_x = cart_center_x + float(robot_on_cart_offset[0])
            cart_full_size = np.array((*XLEROBOT_CART_FOOTPRINT, 0.0), dtype=float)
            cart_center_x = desk_arm_base_x - XLEROBOT_ARM_FORWARD_X
            # ...but never at the price of driving the chassis into the table. The arms
            # bolt to the BACK of the cart, so matching the desk arm's x exactly would
            # push the cart 3 cm past the table edge. Clamp to the table standoff; the
            # arms then sit slightly further back than the desk arm did.
            max_cart_center_x = (
                -(table_full_size[0] + cart_full_size[0]) / 2.0 - float(cart_gap)
            )
            cart_center_x = min(cart_center_x, max_cart_center_x)
            # y=0: the two arms straddle the centreline at +/-0.15, so the desk arm's
            # single-arm y offset does not apply.
            cart_top = np.array([cart_center_x, 0.0, 0.0], dtype=float)
            robot_on_cart_offset = (0.0, 0.0, 0.0)
            # SOARM101Lift hardcodes a NullMount (right for a desk arm); a mobile robot
            # needs its real mobile base or robot0_base_pos has no site to read.
            kwargs.setdefault("base_types", "default")

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
        if self.robot_is_self_supporting:
            # robot_base_pos only drives the head-camera extrinsic, which is defined
            # relative to the ARM mount. For a self-supporting robot the base frame is
            # on the floor, so point this at the midpoint of the two arm bases instead
            # — that keeps the cameras framed as they are for the desk arm.
            self.robot_base_pos = np.array(
                [cart_center_x + XLEROBOT_ARM_FORWARD_X, 0.0, XLEROBOT_ARM_DECK_Z],
                dtype=float,
            )
        super().__init__(*args, reward_shaping=reward_shaping, **kwargs)

    def _add_robot_support_table(self, mujoco_arena):
        # A self-supporting robot brings its own chassis; adding the stand-in table
        # here would bury that chassis inside a second collision box.
        if self.robot_is_self_supporting:
            return
        super()._add_robot_support_table(mujoco_arena)

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

        center, low, high = self._cup_bay_center_and_bounds()
        pad_center = center.copy()
        pad_center[2] = self.coffee_tray_top_z + 0.0015
        pad_half_size = 0.5 * (high - low)
        pad_half_size[2] = 0.0015

        ET.SubElement(
            self.model.worldbody,
            "geom",
            {
                "name": "coffee_success_target_visual",
                "type": "box",
                "pos": " ".join(str(float(v)) for v in pad_center),
                "size": " ".join(str(float(v)) for v in pad_half_size),
                "rgba": "1 1 1 0.8",
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

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
        _, low, high = self._cup_bay_center_and_bounds()
        return bool(np.all(cup_pos >= low) and np.all(cup_pos <= high))

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
        geom_id = self.sim.model.geom_name2id("coffee_success_target_visual")
        self.sim.model.geom_rgba[geom_id] = (
            np.array([0.15, 1.0, 0.15, 0.9]) if success else np.array([1.0, 1.0, 1.0, 0.8])
        )

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
