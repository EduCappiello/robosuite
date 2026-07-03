from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
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
        cube_mass=0.03,
        cube_yaw_range_deg=(0.0, 45.0),
        cube_offset=(-0.17, -0.15, 0.0),
        target_offset=(-0.27, 0.22, 0.0),   # blue X drop-target, aligned to real top cam
        robot_table_offset=(0.182, 0.0, 0.0),
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
        self.target_offset = np.array(target_offset)
        self.robot_table_offset = np.array(robot_table_offset)
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

        xpos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        xpos += self.robot_table_offset
        xpos[2] = self.table_offset[2] + self.robot_table_offset[2]
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

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

        # Blue "X" drop-target decal on the table top (visual only), matching the real scene
        # so the policy isn't shown an out-of-distribution empty target zone. Two crossed thin
        # boxes flat on the surface; position = table_offset + target_offset (world XY).
        import xml.etree.ElementTree as _ET
        _tx = float(self.table_offset[0] + self.target_offset[0])
        _ty = float(self.table_offset[1] + self.target_offset[1])
        _tz = float(self.table_offset[2] + 0.0015)
        for _nm, _ang in (("target_x_a", "0 0 1 0.7853982"), ("target_x_b", "0 0 1 -0.7853982")):
            _ET.SubElement(mujoco_arena.worldbody, "geom", {
                "name": _nm, "type": "box", "size": "0.032 0.0015 0.0001",
                "pos": f"{_tx} {_ty} {_tz}", "axisangle": _ang,
                "rgba": "0.05 0.2 0.85 1", "contype": "0", "conaffinity": "0", "group": "1",
            })

        cube_size = (0.018, 0.018, 0.018)
        cube_volume = 8.0 * cube_size[0] * cube_size[1] * cube_size[2]
        density = float(self.cube_mass / cube_volume)
        self.cube = BoxObject(
            name="cube",
            size_min=cube_size,
            size_max=cube_size,
            density=density,
            rgba=[0.2, 0.2, 0.2, 1],   # default; per-episode colour set live by robot.dress_cube()
            rng=self.rng,
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
                # Randomize only the yaw (rotation about the table normal) in the requested
                # range; re-sampled every reset by _reset_internal -> per-episode orientation.
                rotation=(np.deg2rad(self.cube_yaw_range_deg[0]), np.deg2rad(self.cube_yaw_range_deg[1])),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + self.cube_offset,
                z_offset=0.01,
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

        # "top" — overhead bird's-eye view, looking straight down.
        #   pos[2]: height above world origin — raise to zoom out, lower to zoom in.
        #   pos[0]: shift along world X (positive = toward front of table).
        #   quat: (0.7071, 0, 0, −0.7071) keeps the robot base at the top of
        #         the image; change sign of the last component to mirror L/R.
        mujoco_arena.set_camera(
            camera_name="top",
            pos=np.array([-0.15, 0.02, 1.52]),   # aligned to real top cam (cyan/red overlay)
            quat=np.array([0.7071, 0.0, 0.0, -0.7071]),
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

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
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
