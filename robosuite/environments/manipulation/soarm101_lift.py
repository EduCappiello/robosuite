from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
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
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        cube_mass=0.03,
        cube_offset=(-0.2, -0.2, 0.0),
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
        self.cube_offset = np.array(cube_offset)
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

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube_size = (0.018, 0.018, 0.018)
        cube_volume = 8.0 * cube_size[0] * cube_size[1] * cube_size[2]
        density = float(self.cube_mass / cube_volume)
        self.cube = BoxObject(
            name="cube",
            size_min=cube_size,
            size_max=cube_size,
            density=density,
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
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
            pos=np.array([-0.25, 0.0, 1.70]),   # raised to 1.70 m → robot fits comfortably
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
            pos=np.array([0.05, -0.45, 0.87]),
            quat=np.array([0.6322, 0.6924, 0.2568, 0.2345]),
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
