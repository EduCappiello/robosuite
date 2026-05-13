import numpy as np

import robosuite as suite
from robosuite.robots import FixedBaseRobot


def test_soarm101_robot_loads():
    robot = FixedBaseRobot(robot_type="SOARM101", gripper_type="SO101Gripper")
    robot.load_model()

    assert robot.robot_model.dof == 6
    assert robot.robot_model.naming_prefix == "robot0_"
    assert robot.robot_model.eef_name == {"right": "robot0_gripper"}


def test_soarm101_lift_env_steps():
    env = suite.make(
        env_name="Lift",
        robots=["SOARM101"],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    try:
        env.reset()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, info = env.step(action)

        assert action.shape == env.action_spec[0].shape
        assert isinstance(obs, dict)
    finally:
        env.close()