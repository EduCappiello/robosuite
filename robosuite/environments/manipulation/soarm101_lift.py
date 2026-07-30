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
        cube_mass=0.015,
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
        # sampler. Per-episode colour + mass are applied on the LIVE model by the robot's
        # dress_cube() (see RobosuiteSimFollower) — no env rebuild needed for those.
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

        # Blue "X" drop-target decal on the table top (visual only), matching the real scene
        # so the policy isn't shown an out-of-distribution empty target zone. Two crossed thin
        # boxes flat on the surface; position = table_offset + target_offset (world XY).
        _tx = float(self.table_offset[0] + self.target_offset[0])
        _ty = float(self.table_offset[1] + self.target_offset[1])
        _tz = float(self.table_offset[2] + 0.0015)
        for _nm, _ang in (("target_x_a", "0 0 1 0.7853982"), ("target_x_b", "0 0 1 -0.7853982")):
            ET.SubElement(mujoco_arena.worldbody, "geom", {
                "name": _nm, "type": "box", "size": "0.032 0.0015 0.0001",
                "pos": f"{_tx} {_ty} {_tz}", "axisangle": _ang,
                "rgba": "0.05 0.2 0.85 1", "contype": "0", "conaffinity": "0", "group": "1",
            })

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

        # These values come directly from the block model's receptacle site,
        # base top, front opening, and inner face of its back wall.
        receptacle_local = np.array([-0.0554, 0.0, 0.085])
        self.coffee_target_center = machine_pos + receptacle_local * uniform_scale

        tray_top_local_z = 0.035
        tray_local_x_min = -0.1385
        tray_local_x_max = 0.1035
        cup_radius = float(self.cup_size[0])
        self.coffee_tray_top_z = float(machine_pos[2] + tray_top_local_z * uniform_scale)
        self.coffee_tray_center_x_bounds = np.array(
            [
                machine_pos[0] + tray_local_x_min * uniform_scale + cup_radius,
                machine_pos[0] + tray_local_x_max * uniform_scale - cup_radius,
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
                reference_pos=self.table_offset + np.array([0.055, 0.0, 0.0]),
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

    def _add_robot_support_table(self, mujoco_arena):
        """Add the optional fixed cart-sized table used to support the robot."""
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
