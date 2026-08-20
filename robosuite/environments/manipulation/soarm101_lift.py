from collections import OrderedDict
import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# --- self-supporting (mobile) robots -------------------------------------------
# XLeRobot arrives on its own IKEA RASKOG cart, so it REPLACES the fixed support
# table these tasks build for the desk arm. Keep in sync with XLEROBOT_FRAME_TREE
# in robosuite/models/robots/manipulators/xlerobot_robot.py.
XLEROBOT_CART_FOOTPRINT = (0.392, 0.467)   # chassis x, y from the RASKOG mesh
XLEROBOT_ARM_DECK_Z = 0.8215               # arm-base height on the mounting pads
XLEROBOT_ARM_FORWARD_X = -0.0911           # arms sit BEHIND the chassis centre
XLEROBOT_CUP_HOLDER_CENTER = np.array([0.10480, -0.10830, 0.0], dtype=float)
XLEROBOT_CUP_HOLDER_HALF_SIZE = np.array([0.03500, 0.03500], dtype=float)
XLEROBOT_CUP_HOLDER_SUPPORT_Z = 0.74500
SELF_SUPPORT_TABLE_GAP = 0.01              # chassis-to-table clearance


def robots_bring_their_own_base(robots) -> bool:
    """True if any requested robot is mobile, i.e. it stands on the floor on its own
    chassis and must NOT be perched on the fixed support table."""
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.mobile_robot import MobileRobot

    if robots is None:
        return False
    names = [robots] if isinstance(robots, str) else list(robots)
    return any(
        (cls := ROBOT_CLASS_MAPPING.get(n)) is not None and issubclass(cls, MobileRobot)
        for n in names
    )


class SOARM101Lift(Lift):
    """
    Lift task variant for the SOARM101 arm mounted directly on the table.

    This scenario keeps the standard Lift task geometry and reward structure,
    but defaults to a mount-free SOARM101 setup and lets callers choose the
    cube mass per episode.
    """

    def __init__(
        self,
        robots=("SOARM101",),
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="NullMount",
        initialization_noise=None,   # deterministic reset to init_qpos (no start-pose variance)
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        cube_mass=None,   # None = keep the cup XML's native mass (0.015 kg dry / 0.215 kg water)
        cube_yaw_range_deg=(0.0, 45.0),
        cube_offset=(-0.17, -0.15, 0.0),
        coffee_machine_offset=(0.19, 0.0, 0.0),
        coffee_machine_size=(0.245, 0.277, 0.434),
        include_coffee_machine=True,
        cup_bay_size=(0.150, 0.110),
        cup_size=(0.030, 0.050),
        cup_model_path="objects/coffee_cup_3/model.xml",
        target_offset=(-0.27, 0.22, 0.0),   # blue X drop-target, aligned to real top cam
        robot_table_offset=(0.182, 0.0, 0.0),
        robot_support_table_full_size=None,
        robot_support_table_offset=None,
        robot_support_robot_offset=(0.0, 0.0, 0.0),
        robot_support_table_yaw=0.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.cube_mass = cube_mass
        # Planar (yaw) orientation range for the box, re-sampled every reset by the placement
        # sampler. Per-episode colour + mass can also be applied on the LIVE model by
        # dress_cube() (see scripts/sim_validation on the lerobot side) — no env rebuild needed.
        # cube_mass (when not None) is applied to the compiled model in _setup_references,
        # scaling the cup body's inertia proportionally; the payload knob for estimator
        # validation sweeps (48/102/203 g style).
        self.cube_yaw_range_deg = tuple(cube_yaw_range_deg)
        self.cube_offset = np.array(cube_offset)
        self.coffee_machine_offset = np.array(coffee_machine_offset)
        self.coffee_machine_size = np.array(coffee_machine_size)
        self.include_coffee_machine = bool(include_coffee_machine)
        self.cup_bay_size = tuple(cup_bay_size)
        self.cup_size = tuple(cup_size)
        self.cup_model_path = str(cup_model_path)
        self.target_offset = np.array(target_offset)
        self.robot_table_offset = np.array(robot_table_offset)
        self.robot_support_table_full_size = (
            None if robot_support_table_full_size is None else np.array(robot_support_table_full_size, dtype=float)
        )
        self.robot_support_table_offset = (
            None if robot_support_table_offset is None else np.array(robot_support_table_offset, dtype=float)
        )
        self.robot_support_robot_offset = np.array(robot_support_robot_offset, dtype=float)
        self.robot_support_table_yaw = float(robot_support_table_yaw)
        if (self.robot_support_table_full_size is None) != (self.robot_support_table_offset is None):
            raise ValueError("robot support table size and offset must be provided together")
        # Manipulation tasks keep XLeRobot parked: retain its three base channels for
        # 17-DoF dataset parity, but constrain them so contact cannot move the chassis.
        self.robot_is_self_supporting = robots_bring_their_own_base(robots)
        if self.robot_is_self_supporting and base_types == "NullMount":
            names = [robots] if isinstance(robots, str) else list(robots)
            base_types = "LockedNullMobileBase" if "XLeRobot" in names else "default"
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            table_full_size=table_full_size,
            table_friction=table_friction,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        super(Lift, self)._load_model()

        if self.robot_support_table_offset is None:
            xpos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
            xpos += self.robot_table_offset
            xpos[2] = self.table_offset[2] + self.robot_table_offset[2]
        else:
            yaw = self.robot_support_table_yaw
            rot_z = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            xpos = self.robot_support_table_offset + rot_z @ self.robot_support_robot_offset
        if self.robot_is_self_supporting:
            xpos = self._self_supporting_base_xpos(xpos)
        self.robot_world_base_pos = np.array(xpos, dtype=float)
        self.robots[0].robot_model.set_base_xpos(xpos)
        if self.robot_support_table_offset is not None:
            self.robots[0].robot_model.set_base_ori(
                np.array([0.0, 0.0, self.robot_support_table_yaw])
            )

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_robot_support_table(mujoco_arena)

        # Quick test: beige background (floor plane + skybox void) instead of wood/gray.
        _beige = "0.82 0.71 0.55 1"
        _floor = getattr(mujoco_arena, "floor", None)
        if _floor is not None:
            _floor.set("rgba", _beige)
            _floor.attrib.pop("material", None)
        # Walls (the gray behind the arm in the corner view) and skybox -> beige.
        for _mat in mujoco_arena.asset.findall(".//material[@name='walls_mat']"):
            _mat.set("rgba", _beige)
            _mat.attrib.pop("texture", None)
        for _tex in mujoco_arena.asset.findall(".//texture[@type='skybox']"):
            _tex.set("builtin", "flat")
            _tex.set("rgb1", "0.82 0.71 0.55")
            _tex.set("rgb2", "0.82 0.71 0.55")
            _tex.attrib.pop("file", None)

        # A matte cool-gray tabletop gives the cup base a clear silhouette.
        # Keep the visual and collision geometry unchanged; only adjust appearance.
        for _mat in mujoco_arena.asset.findall(".//material[@name='table_ceramic']"):
            _mat.set("rgba", "0.72 0.76 0.78 1")
            _mat.set("reflectance", "0.0")
            _mat.set("shininess", "0.05")
            _mat.set("specular", "0.05")
            _mat.attrib.pop("texture", None)
        for _light in mujoco_arena.worldbody.findall(".//light"):
            _light.set("castshadow", "false")

        # Keep the real-scene blue X for the desk-mounted SO101 setup only.
        # XLeRobot does not have this decal on its cart, so it must also stay out
        # of recorded observations rather than being hidden in the viewer alone.
        if not self.robot_is_self_supporting:
            _tx = float(self.table_offset[0] + self.target_offset[0])
            _ty = float(self.table_offset[1] + self.target_offset[1])
            _tz = float(self.table_offset[2] + 0.0015)
            for _nm, _ang in (
                ("target_x_a", "0 0 1 0.7853982"),
                ("target_x_b", "0 0 1 -0.7853982"),
            ):
                ET.SubElement(
                    mujoco_arena.worldbody,
                    "geom",
                    {
                        "name": _nm,
                        "type": "box",
                        "size": "0.032 0.0015 0.0001",
                        "pos": f"{_tx} {_ty} {_tz}",
                        "axisangle": _ang,
                        "rgba": "0.05 0.2 0.85 1",
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "1",
                    },
                )

        # Original block-built XLeRobot coffee machine. Its source dimensions
        # are 24.5 x 27.7 x 43.4 cm (width Y, depth X, height Z).
        cm_width, cm_depth, cm_height = [float(v) for v in self.coffee_machine_size]
        cm_cx = float(self.table_offset[0] + self.coffee_machine_offset[0])
        cm_cy = float(self.table_offset[1] + self.coffee_machine_offset[1])
        cm_z0 = float(self.table_offset[2])
        source_width, source_depth, source_height = 0.245, 0.277, 0.434
        uniform_scale = cm_height / source_height
        requested_width = source_width * uniform_scale
        requested_depth = source_depth * uniform_scale
        if not np.allclose(
            [cm_width, cm_depth],
            [requested_width, requested_depth],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "coffee_machine_size must preserve the block machine aspect ratio; "
                f"expected width/depth {(requested_width, requested_depth)} for height {cm_height}"
            )
        self.coffee_machine_scale = float(uniform_scale)

        # The block model origin is the footprint center on the table and local
        # -X already faces the robot, so no world rotation is needed.
        machine_pos = np.array([cm_cx, cm_cy, cm_z0])
        machine_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.coffee_machine = MujocoXMLObject(
            fname=xml_path_completion("objects/coffee_machine_block_xlerobot/model.xml"),
            name="coffee_machine",
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=uniform_scale,
        )
        self.coffee_machine.get_obj().set("pos", " ".join(str(float(v)) for v in machine_pos))
        self.coffee_machine.get_obj().set("quat", " ".join(str(float(v)) for v in machine_quat))

        # Physical white pad inside the machine. The pad is 10 cm in diameter,
        # 3 mm thick, and its front edge sits 3.5 cm behind the machine front.
        machine_front_local_x = -0.1385
        pad_front_clearance = 0.035
        pad_radius_local = 0.050
        pad_half_height_local = 0.0015
        pad_center_local_x = machine_front_local_x + pad_front_clearance + pad_radius_local
        pad_center_local_y = 0.02375
        pad_top_local_z = 0.035 + 2.0 * pad_half_height_local
        cup_radius = float(self.cup_size[0])
        cup_half_height = float(self.cup_size[1])

        self.coffee_pad_radius = float(pad_radius_local * uniform_scale)
        self.coffee_pad_center = machine_pos + np.array(
            [pad_center_local_x, pad_center_local_y, pad_top_local_z]
        ) * uniform_scale
        self.coffee_tray_top_z = float(self.coffee_pad_center[2])
        self.coffee_target_center = self.coffee_pad_center.copy()
        self.coffee_target_center[2] += cup_half_height + 0.002

        center_tolerance = max(0.0, self.coffee_pad_radius - cup_radius)
        self.coffee_tray_center_x_bounds = np.array(
            [
                self.coffee_pad_center[0] - center_tolerance,
                self.coffee_pad_center[0] + center_tolerance,
            ],
            dtype=float,
        )
        # RoboCasa coffee_cup_3, scaled from 6.46 x 6.46 x 10.60 cm to 6 x 6 x 10 cm.
        # Its collision shell is decomposed into 16 convex meshes for stable grasp contacts.
        self.cube = MujocoXMLObject(
            fname=xml_path_completion(self.cup_model_path),
            name="cup",
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=[0.928656805, 0.928667580, 0.943260902],
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.00, 0.00],
                y_range=[-0.00, 0.00],
                rotation=(0.0, 0.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                # cube_offset is the caller's placement knob (scene-match / payload
                # harness). The PnP task envs ignore this sampler and set the cup
                # pose directly from their own cup_start_offset at reset.
                reference_pos=self.table_offset + self.cube_offset,
                z_offset=0.005,
                rng=self.rng,
            )

        # Operator-facing view: camera from the −X side looking at the arm front.
        mujoco_arena.set_camera(
            camera_name="operator_view",
            pos=np.array([-1.5, 0.0, 1.45]),
            quat=np.array([0.613, 0.353, -0.353, -0.612]),
        )

        # ── Recording cameras (match real-world lerobot camera rig at 640×480) ──
        # ┌──────────────────────────────────────────────────────────────────────┐
        # │  To tune cameras manually edit only the pos / quat values below.    │
        # │  pos = [x, y, z] in world metres.                                   │
        # │  quat = [w, x, y, z] (MuJoCo convention).                           │
        # │  Recompute quat with demo_soarm101_teleop.py —the two camera windows │
        # │  update in real-time so you can iterate quickly.                     │
        # └──────────────────────────────────────────────────────────────────────┘

        # "top" is the RealSense mounted at the XLeRobot neck/head location.
        # It looks from behind the robot toward the cup and coffee-machine workspace.
        mujoco_arena.set_camera(
            camera_name="top",
            pos=np.array([-0.78, -0.25, 1.26065]),
            quat=np.array([0.6540660, 0.4531726, -0.3449356, -0.4978471]),
            # RealSense-like RGB projection. Simulated depth uses the same
            # pinhole projection and is therefore pixel-aligned with RGB.
            camera_attribs={"fovy": "42.5"},
        )

        # Rear workspace view: behind the robot, looking forward toward the
        # cup, gripper workspace, and coffee-machine target region.
        mujoco_arena.set_camera(
            camera_name="rear",
            pos=np.array([-0.78, -0.25, 1.26065]),
            quat=np.array([0.6540660, 0.4531726, -0.3449356, -0.4978471]),
            camera_attribs={"fovy": "34"},
        )
        # "corner" — right-front corner of the table, 4 cm above the surface.
        #   Table surface ≈ z = 0.83 m  →  4 cm up = 0.87 m.
        #   pos[0]: positive X = front edge of table.
        #   pos[1]: negative Y = right side of table (from robot's perspective).
        #   quat: points from (0.05, −0.45, 0.87) toward the robot workspace;
        #         recomputed via cross-product rotation (see git log for formula).
        mujoco_arena.set_camera(
            camera_name="corner",
            # Aligned to real corner cam (cyan/red overlay). Pos lowered to 0.84; quat pitched
            # up ~6 deg (camera points a little upward -> lower horizon, matching the real view).
            pos=np.array([0.07, -0.40, 0.84]),
            quat=np.array([0.5951, 0.7245, 0.2687, 0.2207]),
        )

        task_objects = [self.cube]
        if self.include_coffee_machine:
            task_objects.insert(0, self.coffee_machine)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=task_objects,
        )
        if self.include_coffee_machine:
            # ManipulationTask merges an object's body and assets. Merge the
            # block machine's button sensors and self-contact exclusion too.
            self.model.merge(self.coffee_machine, merge_body=None)

    def _setup_references(self):
        super()._setup_references()
        # Apply the requested payload mass to the cup body on the compiled model,
        # scaling its rotational inertia by the same ratio. Idempotent across resets.
        if self.cube_mass is not None:
            current = float(self.sim.model.body_mass[self.cube_body_id])
            ratio = self.cube_mass / current
            self.sim.model.body_mass[self.cube_body_id] *= ratio
            self.sim.model.body_inertia[self.cube_body_id] *= ratio

        model_token = id(self.sim.model)
        if getattr(self, "_cup_visual_model_token", None) != model_token:
            self._cup_visual_model_token = model_token
            self._cup_visual_geom_ids = np.asarray(
                [self.sim.model.geom_name2id(name) for name in self.cube.visual_geoms],
                dtype=int,
            )
            self._cup_visual_base_rgba = self.sim.model.geom_rgba[
                self._cup_visual_geom_ids
            ].copy()

    def _sample_cup_xy_jitter(self, max_radius_m):
        """Sample a uniform point in a disk while respecting deterministic resets."""
        max_radius_m = float(max_radius_m)
        if self.deterministic_reset or max_radius_m <= 0.0:
            return np.zeros(2, dtype=float)
        radius = max_radius_m * np.sqrt(self.rng.uniform())
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        return radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)

    def _sample_cup_yaw_jitter_deg(self, max_abs_deg):
        max_abs_deg = float(max_abs_deg)
        if self.deterministic_reset or max_abs_deg <= 0.0:
            return 0.0
        return float(self.rng.uniform(-max_abs_deg, max_abs_deg))

    def _randomize_cup_visual_color(self, max_fraction):
        """Apply a subtle RGB tint to cup visual geoms without changing physics."""
        max_fraction = float(max_fraction)
        tint = np.ones(3, dtype=float)
        if not self.deterministic_reset and max_fraction > 0.0:
            tint = self.rng.uniform(1.0 - max_fraction, 1.0 + max_fraction, size=3)

        rgba = self._cup_visual_base_rgba.copy()
        rgba[:, :3] = np.clip(rgba[:, :3] * tint, 0.0, 1.0)
        self.sim.model.geom_rgba[self._cup_visual_geom_ids] = rgba

    def _task_manipulation_arm(self):
        """Return the arm used by cup tasks on single- and dual-arm robots."""
        arms = tuple(self.robots[0].arms)
        return "left" if "left" in arms else arms[0]

    def _task_gripper_tip_pos(self):
        """Return the physical fingertip-center site, not the adapter origin."""
        robot = self.robots[0]
        arm = self._task_manipulation_arm()
        raw_site_name = f"{arm}_grip_site" if len(robot.arms) > 1 else "grip_site"
        site_name = robot.robot_model.correct_naming(raw_site_name)
        return np.array(self.sim.data.get_site_xpos(site_name), dtype=float)

    def _task_gripper_contacts_object(self, object_model):
        """Check any active-jaw contact instead of the empty SO101 fingerpad groups."""
        robot = self.robots[0]
        arm = self._task_manipulation_arm()
        raw_prefix = f"{arm}_" if len(robot.arms) > 1 else ""
        geom_prefix = robot.robot_model.correct_naming(raw_prefix)
        contact_geoms = []
        for geom_id in range(self.sim.model.ngeom):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is None or not geom_name.startswith(geom_prefix):
                continue
            if (
                self.sim.model.geom_contype[geom_id] == 0
                and self.sim.model.geom_conaffinity[geom_id] == 0
            ):
                continue
            local_name = geom_name[len(geom_prefix):]
            if local_name.startswith(("static_finger", "moving_jaw")):
                contact_geoms.append(geom_name)
        return bool(
            contact_geoms
            and self.check_contact(contact_geoms, object_model.contact_geoms)
        )

    def _self_supporting_base_xpos(self, desk_arm_xpos):
        """Where to park a robot that arrives on its own chassis (XLeRobot).

        `desk_arm_xpos` is where the single desk arm's BASE would have gone. A mobile
        robot must be placed so its ARM BASES land there instead — its arms are offset
        from the chassis centre (XLeRobot's sit 91 mm BEHIND it, on the top plate's
        mounting pads), so matching chassis-to-chassis would put the arms in a
        completely different place than the task was designed around.

        The result is then clamped so the chassis cannot intersect the main table:
        the arms bolt to the rear of the cart, so honouring the desk arm's x exactly
        would drive the cart through the table edge. When clamped, the arms end up
        slightly further back than the desk arm was, which costs a little reach —
        that is a real consequence of the mounting, not a bug.
        """
        yaw = self.robot_support_table_yaw
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        cart_centre = np.asarray(desk_arm_xpos, dtype=float) - XLEROBOT_ARM_FORWARD_X * forward

        # Keep the chassis clear of the main table along the approach axis.
        table_near = -self.table_full_size[0] / 2.0
        max_centre_x = table_near - XLEROBOT_CART_FOOTPRINT[0] / 2.0 - SELF_SUPPORT_TABLE_GAP
        cart_centre[0] = min(cart_centre[0], max_centre_x)
        # Two arms straddle the centreline, so the desk arm's lateral offset does not
        # apply; and the base frame origin is on the floor.
        cart_centre[1] = 0.0
        cart_centre[2] = 0.0
        return cart_centre

    def _add_robot_support_table(self, mujoco_arena):
        """Add the optional fixed cart-sized table used to support the robot."""
        # A self-supporting robot brings its own chassis; building the stand-in table
        # here would bury that chassis inside a second collision box.
        if self.robot_is_self_supporting:
            return
        if self.robot_support_table_offset is None:
            return

        full_size = self.robot_support_table_full_size
        half_size = full_size / 2.0
        top = self.robot_support_table_offset
        tabletop_center = top - np.array([0.0, 0.0, half_size[2]])
        body = ET.SubElement(
            mujoco_arena.worldbody,
            "body",
            {
                "name": "robot_support_table",
                "pos": " ".join(str(float(v)) for v in tabletop_center),
                "quat": (
                    f"{np.cos(self.robot_support_table_yaw / 2.0)} 0 0 "
                    f"{np.sin(self.robot_support_table_yaw / 2.0)}"
                ),
            },
        )
        common = {
            "type": "box",
            "pos": "0 0 0",
            "size": " ".join(str(float(v)) for v in half_size),
        }
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": "robot_support_table_collision",
                "friction": " ".join(str(float(v)) for v in self.table_friction),
                "group": "0",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": "robot_support_table_visual",
                "material": "table_ceramic",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

        leg_radius = 0.018
        leg_half_height = max((top[2] - full_size[2]) / 2.0, 0.001)
        leg_inset = 0.035
        yaw = self.robot_support_table_yaw
        rot_xy = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        for index, sx, sy in ((0, 1, 1), (1, 1, -1), (2, -1, 1), (3, -1, -1)):
            leg_xy = top[:2] + rot_xy @ np.array(
                [sx * (half_size[0] - leg_inset), sy * (half_size[1] - leg_inset)]
            )
            leg_world = np.array(
                [
                    leg_xy[0],
                    leg_xy[1],
                    leg_half_height,
                ]
            )
            ET.SubElement(
                mujoco_arena.worldbody,
                "geom",
                {
                    "name": f"robot_support_leg{index}_collision",
                    "type": "cylinder",
                    "pos": " ".join(str(float(v)) for v in leg_world),
                    "size": f"{leg_radius} {leg_half_height}",
                    "friction": " ".join(str(float(v)) for v in self.table_friction),
                    "group": "0",
                },
            )
            ET.SubElement(
                mujoco_arena.worldbody,
                "geom",
                {
                    "name": f"robot_support_leg{index}_visual",
                    "type": "cylinder",
                    "pos": " ".join(str(float(v)) for v in leg_world),
                    "size": f"{leg_radius} {leg_half_height}",
                    "material": "table_legs_metal",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )

    def _setup_observables(self):
        observables = super(Lift, self)._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super(Lift, self)._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
