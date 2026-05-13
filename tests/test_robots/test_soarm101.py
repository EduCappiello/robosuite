import numpy as np

import robosuite as suite
from robosuite.robots import FixedBaseRobot


def test_soarm101_robot_loads():
    robot = FixedBaseRobot(robot_type="SOARM101", gripper_type="SO101Gripper")
    robot.load_model()

    assert robot.robot_model.dof == 6
    assert robot.robot_model.naming_prefix == "robot0_"
    assert robot.robot_model.eef_name == {"right": "robot0_gripper"}
    assert robot.gripper["right"].dof == 1
    assert robot.gripper["right"].joints == ["robot0_gripper"]
    assert robot.gripper["right"].actuators == ["robot0_gripper"]


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
        assert env.robots[0].gripper_joints["right"] == ["robot0_gripper"]
        assert len(env.robots[0]._ref_joint_gripper_actuator_indexes["right"]) == 1

        env.reset()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, info = env.step(action)

        assert action.shape == env.action_spec[0].shape
        assert isinstance(obs, dict)
    finally:
        env.close()


def test_soarm101_shoulder_pan_moves_with_collisions_enabled():
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
        sim = env.sim
        qpos_addr = sim.model.get_joint_qpos_addr("robot0_shoulder_pan")
        actuator_id = sim.model.actuator_name2id("robot0_shoulder_pan")

        q0 = float(sim.data.qpos[qpos_addr])
        for _ in range(300):
            sim.data.ctrl[:] = 0.0
            sim.data.ctrl[actuator_id] = 3.35
            sim.step()
        q1 = float(sim.data.qpos[qpos_addr])

        # Shoulder pan should move significantly when directly actuated.
        assert abs(q1 - q0) > 0.2
    finally:
        env.close()