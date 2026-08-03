import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Room128Arena
from robosuite.models.tasks import ManipulationTask


class Room128(ManipulationEnv):
    """The robot standing in Room 128, with nothing else in it.

    Room 128 is the classroom the real XLeRobot coffee dataset was recorded in
    (every episode in IntelligentDecisionLab/xlerobot-coffee-real lives under
    ``room-128/``). This env is the room plus the robot and no task objects: it is
    the drive-around sandbox, and the foundation the real dataset's ``t4_navigate``
    task needs — that task cannot be expressed on a tabletop arena, which is why
    the cupPnP set has no task 4.

    Walls and furniture COLLIDE, so the base can be driven into them.

    Reward is always 0; there is no success condition. It exists to place and move
    the robot, not to score it.

    Args:
        robot_start_pos (3-array): where to park the base, in room coordinates.
            The room is centred on the origin: x +/-1.83, y +/-2.47, floor at z=0.
        robot_start_yaw (float): base yaw in radians.
    """

    def __init__(
        self,
        robots=("XLeRobot",),
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise=None,
        robot_start_pos=(0.0, -0.6, 0.0),
        robot_start_yaw=0.0,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        control_freq=20,
        **kwargs,
    ):
        self.robot_start_pos = np.array(robot_start_pos, dtype=float)
        self.robot_start_yaw = float(robot_start_yaw)
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            **kwargs,
        )

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(self.robot_start_pos)
        if self.robot_start_yaw:
            self.robots[0].robot_model.set_base_ori(
                np.array([0.0, 0.0, self.robot_start_yaw])
            )
        mujoco_arena = Room128Arena()
        mujoco_arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )

    def _check_success(self):
        return False
