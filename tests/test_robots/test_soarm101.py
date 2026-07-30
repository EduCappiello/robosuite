import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import robosuite as suite
from robosuite.robots import FixedBaseRobot
from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift
from robosuite.environments.manipulation.soarm101_PnPcup import SOARM101PnPCup
from robosuite.environments.manipulation.cup_pnp_task1 import cupPnP_task1
from robosuite.environments.manipulation.cup_pnp_task3 import cupPnP_task3
from robosuite.environments.manipulation.cup_pnp_task5 import cupPnP_task5


ASSET_ROOT = Path(__file__).resolve().parents[2] / "robosuite" / "models" / "assets"


@pytest.mark.parametrize(
    ("env_name", "geom_names", "reset_color"),
    [
        ("SOARM101PnPCup", ("coffee_success_target_visual",), [1.0, 1.0, 1.0, 0.8]),
        ("cupPnP_task1", ("coffee_success_target_visual",), [1.0, 1.0, 1.0, 0.8]),
        ("cupPnP_task3", ("target_x_a", "target_x_b"), [0.05, 0.2, 0.85, 1.0]),
        ("cupPnP_task5", ("target_x_a", "target_x_b"), [0.05, 0.2, 0.85, 1.0]),
    ],
)
def test_pnp_success_visual_resets_between_episodes(env_name, geom_names, reset_color):
    env = suite.make(
        env_name=env_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        control_freq=20,
    )

    try:
        env.reset()
        update_visual = getattr(
            env,
            "_update_success_target_visual",
            getattr(env, "_update_target_visual", None),
        )
        assert update_visual is not None
        update_visual(True)
        for geom_name in geom_names:
            geom_id = env.sim.model.geom_name2id(geom_name)
            assert np.allclose(env.sim.model.geom_rgba[geom_id], [0.15, 1.0, 0.15, 0.9])

        env.reset()
        assert env._success_counter == 0
        for geom_name in geom_names:
            geom_id = env.sim.model.geom_name2id(geom_name)
            assert np.allclose(env.sim.model.geom_rgba[geom_id], reset_color)
    finally:
        env.close()


def test_soarm101_robot_loads():
    robot = FixedBaseRobot(robot_type="SOARM101", gripper_type="SO101Gripper")
    robot.load_model()

    assert robot.robot_model.dof == 5  # arm only; gripper DOF is in SO101Gripper
    assert robot.robot_model.naming_prefix == "robot0_"
    assert robot.robot_model.eef_name == {"right": "robot0_gripper"}
    assert robot.gripper["right"].dof == 1
    assert robot.gripper["right"].joints == ["robot0_gripper"]
    assert robot.gripper["right"].actuators == ["robot0_gripper"]


def test_so101_gripper_contact_is_stiff_and_force_limited():
    gripper_root = ET.parse(ASSET_ROOT / "grippers" / "so101_gripper.xml").getroot()
    arm_root = ET.parse(
        ASSET_ROOT / "robots" / "SOARM101" / "SO101" / "soarm_with_sensor.xml"
    ).getroot()

    actuator = gripper_root.find(".//actuator/position[@name='gripper']")
    assert actuator is not None
    assert np.fromstring(actuator.get("forcerange"), sep=" ") == pytest.approx([-1.2, 1.2])

    moving_geoms = [
        geom
        for geom in gripper_root.findall(".//geom")
        if (geom.get("name") or "").startswith("moving_jaw_")
        and "_col_" in (geom.get("name") or "")
    ]
    static_geoms = [
        geom
        for geom in arm_root.findall(".//geom")
        if (geom.get("name") or "").startswith("static_finger_")
    ]
    assert len(moving_geoms) == 12
    assert len(static_geoms) == 15

    for geom in moving_geoms + static_geoms:
        assert np.fromstring(geom.get("solimp"), sep=" ") == pytest.approx(
            [0.995, 0.999, 0.001]
        )
        assert np.fromstring(geom.get("solref"), sep=" ") == pytest.approx([0.005, 1.0])

    for cup_dir in ("coffee_cup_3", "coffee_cup_3_water"):
        cup_root = ET.parse(ASSET_ROOT / "objects" / cup_dir / "model.xml").getroot()
        collision_default = cup_root.find(".//default[@class='collision']/geom")
        assert collision_default is not None
        assert np.fromstring(collision_default.get("solimp"), sep=" ") == pytest.approx(
            [0.995, 0.999, 0.001]
        )


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


def test_soarm101_custom_lift_uses_null_mount_and_custom_mass():
    env = SOARM101Lift(
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        cube_mass=0.12,
        cube_offset=(0.05, -0.02, 0.0),
    )

    try:
        assert env.robots[0].robot_model.base.__class__.__name__ == "NullMount"
        expected_density = 0.12 / (8.0 * 0.018 * 0.018 * 0.018)
        assert np.isclose(env.cube.density, expected_density)
        assert np.allclose(env.placement_initializer.reference_pos, env.table_offset + env.cube_offset)

        env.reset()
        action = np.zeros(env.action_spec[0].shape)
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)
        assert action.shape == env.action_spec[0].shape
    finally:
        env.close()


def test_cup_pnp_task1_mounts_robot_on_cart_sized_support():
    env = suite.make(
        env_name="cupPnP_task1",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    try:
        env.reset()

        assert isinstance(env, cupPnP_task1)
        assert env.cup_model_path == "objects/coffee_cup_3/model.xml"
        assert np.isclose(env.sim.model.body_mass[env.cube_body_id], 0.015)
        assert np.allclose(env.cart_full_size, [0.35, 0.45, 0.04])
        assert np.isclose(env.cart_top[2], 0.77)
        assert np.isclose(env.table_offset[2], 0.865)
        assert np.isclose(env.cart_top[0] + env.cart_full_size[0] / 2.0, -0.41)
        assert np.isclose(-env.table_full_size[0] / 2.0, -0.40)

        support_geom_id = env.sim.model.geom_name2id("robot_support_table_collision")
        assert np.allclose(env.sim.model.geom_size[support_geom_id], env.cart_full_size / 2.0)

        root_body_id = env.sim.model.body_name2id(env.robots[0].robot_model.root_body)
        expected_robot_pos = env.cart_top + np.array([-0.07, 0.10, 0.0])
        assert np.allclose(env.sim.data.body_xpos[root_body_id], expected_robot_pos)

        assert np.allclose(env.coffee_machine_offset, [-0.19, 0.0, 0.0])
        assert np.allclose(env.cup_start_offset, [-0.37, 0.22, 0.0])
        assert np.allclose(env._cup_pos()[:2], [-0.37, 0.22])
    finally:
        env.close()


def test_cup_pnp_task1_reuses_all_pnp_task_logic_and_sensors():
    assert cupPnP_task1.__bases__ == (SOARM101Lift,)
    assert not issubclass(cupPnP_task1, SOARM101PnPCup)

    common = {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "control_freq": 20,
    }
    reference = suite.make(env_name="SOARM101PnPCup", **common)
    positioned = suite.make(env_name="cupPnP_task1", **common)
    try:
        reference_obs = reference.reset()
        positioned_obs = positioned.reset()

        assert set(positioned_obs) == set(reference_obs)
        assert {
            key: np.asarray(value).shape for key, value in positioned_obs.items()
        } == {
            key: np.asarray(value).shape for key, value in reference_obs.items()
        }
        for attr in (
            "coffee_target_half_size",
            "cup_start_yaw_deg",
            "success_hold_steps",
            "cup_max_tilt_deg",
            "cup_failure_tilt_deg",
            "reward_shaping",
        ):
            assert np.allclose(getattr(positioned, attr), getattr(reference, attr))

        reference_robot_id = reference.sim.model.body_name2id(reference.robots[0].robot_model.root_body)
        positioned_robot_id = positioned.sim.model.body_name2id(positioned.robots[0].robot_model.root_body)
        reference_robot_pos = reference.sim.data.body_xpos[reference_robot_id]
        positioned_robot_pos = positioned.sim.data.body_xpos[positioned_robot_id]
        for camera_name in ("top", "rear"):
            reference_camera_id = reference.sim.model.camera_name2id(camera_name)
            positioned_camera_id = positioned.sim.model.camera_name2id(camera_name)
            assert np.allclose(
                positioned.sim.data.cam_xpos[positioned_camera_id] - positioned_robot_pos,
                reference.sim.data.cam_xpos[reference_camera_id] - reference_robot_pos,
            )
            assert np.allclose(
                positioned.sim.data.cam_xmat[positioned_camera_id],
                reference.sim.data.cam_xmat[reference_camera_id],
            )
            assert np.isclose(
                positioned.sim.model.cam_fovy[positioned_camera_id],
                reference.sim.model.cam_fovy[reference_camera_id],
            )

        for task in (reference, positioned):
            center, _, _ = task._cup_bay_center_and_bounds()
            task.sim.data.set_joint_qpos(
                task.cube.joints[0],
                np.concatenate([center, np.array([1.0, 0.0, 0.0, 0.0])]),
            )
            task.sim.forward()
            for _ in range(task.success_hold_steps):
                success = task._check_success()
            assert success
            assert task.task_success

            cup_qpos = task.sim.data.get_joint_qpos(task.cube.joints[0]).copy()
            tilt_rad = np.deg2rad(85.0)
            cup_qpos[3:] = [np.cos(tilt_rad / 2), np.sin(tilt_rad / 2), 0.0, 0.0]
            task.sim.data.set_joint_qpos(task.cube.joints[0], cup_qpos)
            task.sim.forward()
            assert task.task_failed
    finally:
        positioned.close()
        reference.close()


def test_cup_pnp_task3_starts_in_machine_and_targets_support_table(monkeypatch):
    env = suite.make(
        env_name="cupPnP_task3",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    try:
        env.reset()
        assert isinstance(env, cupPnP_task3)
        assert env.cup_model_path == "objects/coffee_cup_3_water/model.xml"
        assert np.isclose(env.sim.model.body_mass[env.cube_body_id], 0.215)

        coffee_start, _, _ = env._cup_bay_center_and_bounds()
        assert np.allclose(env._cup_pos(), coffee_start)

        marker_pos = env._cart_target_pos()
        marker_pos[2] = env.cart_top[2] + 0.0015
        for geom_name in ("target_x_a", "target_x_b"):
            geom_id = env.sim.model.geom_name2id(geom_name)
            assert np.allclose(env.sim.model.geom_pos[geom_id], marker_pos)

        assert marker_pos[0] > env.cart_top[0]
        assert marker_pos[1] < env.cart_top[1]

        # Success is the complete support-table surface, not just the X.
        placed_pos = env.cart_top + np.array([0.05, -0.05, env.cup_size[1]])
        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([placed_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.forward()
        assert env._cup_on_cart()

        hovering_pos = placed_pos.copy()
        hovering_pos[2] += 0.05
        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([hovering_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.forward()
        assert not env._cup_on_cart()

        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([placed_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
        env.sim.forward()
        assert env._cup_is_upright()
        assert env._cup_is_released()
        assert env._gripper_is_clear()
        assert env._cup_is_stable()

        with monkeypatch.context() as patch:
            patch.setattr(env, "_check_grasp", lambda **kwargs: True)
            assert not env._cup_is_released()

        original_clearance = env.gripper_clearance_m
        env.gripper_clearance_m = 10.0
        assert not env._gripper_is_clear()
        env.gripper_clearance_m = original_clearance

        env.sim.data.set_joint_qvel(env.cube.joints[0], np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
        env.sim.forward()
        assert not env._cup_is_stable()

        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
        env.sim.forward()
        for gate in ("_cup_is_released", "_gripper_is_clear", "_cup_is_stable"):
            env._success_counter = 1
            with monkeypatch.context() as patch:
                patch.setattr(env, gate, lambda: False)
                assert not env._check_success()
                assert env._success_counter == 0

        for _ in range(env.success_hold_steps):
            success = env._check_success()
        assert success
        assert env.task_success
    finally:
        env.close()


def test_cup_pnp_task5_rotated_cart_start_and_main_table_success(monkeypatch):
    env = suite.make(
        env_name="cupPnP_task5",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    try:
        env.reset()
        assert isinstance(env, cupPnP_task5)
        assert np.isclose(env.table_offset[2], 0.80)
        assert np.isclose(env.cart_top[2], 0.77)
        assert np.isclose(env.layout_yaw_rad, 3.0 * np.pi / 2.0)
        assert np.isclose(env.sim.model.body_mass[env.cube_body_id], 0.215)

        body_names = [
            env.sim.model.body_id2name(body_id)
            for body_id in range(env.sim.model.nbody)
        ]
        assert not any(name and "coffee_machine" in name for name in body_names)

        expected_rotation = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        support_body_id = env.sim.model.body_name2id("robot_support_table")
        assert np.allclose(
            env.sim.data.body_xmat[support_body_id].reshape(3, 3),
            expected_rotation,
            atol=1e-6,
        )

        robot_body_id = env.sim.model.body_name2id(env.robots[0].robot_model.root_body)
        assert np.allclose(
            env.sim.data.body_xmat[robot_body_id].reshape(3, 3),
            expected_rotation,
            atol=1e-6,
        )
        expected_robot_pos = env.cart_top + expected_rotation @ np.array([-0.07, 0.10, 0.0])
        assert np.allclose(env.sim.data.body_xpos[robot_body_id], expected_robot_pos)

        start_pos = env._cart_start_pos()
        assert np.allclose(env._cup_pos(), start_pos)
        marker_pos = start_pos.copy()
        marker_pos[2] = env.cart_top[2] + 0.0015
        for geom_name in ("target_x_a", "target_x_b"):
            geom_id = env.sim.model.geom_name2id(geom_name)
            assert np.allclose(env.sim.model.geom_pos[geom_id], marker_pos)

        placed_pos = env._main_table_target_pos()
        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([placed_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
        env.sim.forward()
        assert env._cup_on_main_table()
        assert env._cup_is_upright()
        assert env._cup_is_released()
        assert env._gripper_is_clear()
        assert env._cup_is_stable()

        with monkeypatch.context() as patch:
            patch.setattr(env, "_check_grasp", lambda **kwargs: True)
            assert not env._cup_is_released()
            assert not env._check_success()

        original_clearance = env.gripper_clearance_m
        env.gripper_clearance_m = 10.0
        assert not env._gripper_is_clear()
        assert not env._check_success()
        env.gripper_clearance_m = original_clearance

        env.sim.data.set_joint_qvel(env.cube.joints[0], np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
        env.sim.forward()
        assert not env._cup_is_stable()
        assert not env._check_success()

        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
        env.sim.forward()
        for gate in ("_cup_is_released", "_gripper_is_clear", "_cup_is_stable"):
            env._success_counter = 1
            with monkeypatch.context() as patch:
                patch.setattr(env, gate, lambda: False)
                assert not env._check_success()
                assert env._success_counter == 0

        for _ in range(env.success_hold_steps):
            success = env._check_success()
        assert success
        assert env.task_success
    finally:
        env.close()
