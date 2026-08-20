"""
Integrity tests for the 17-DoF XLeRobot robosuite model (M1 of the sim port):
part classification, positional right/left arm split, per-arm gripper namespaces,
frame-tree drift guard against XLEROBOT_FRAME_TREE, init-pose validity (the
joint-limit lesson from test_external_gt), wrist FT sensors, and a 17-dim
action-space smoke roll in a Lift env.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import robosuite as suite
from robosuite.models.robots.manipulators.xlerobot_robot import XLEROBOT_FRAME_TREE, XLeRobot
from robosuite.robots import ROBOT_CLASS_MAPPING, WheeledRobot

ASSET_ROOT = Path(__file__).resolve().parents[2] / "robosuite" / "models" / "assets"
XLEROBOT_XML = ASSET_ROOT / "robots" / "XLeRobot" / "robot.xml"

ARM_JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def test_xlerobot_registration():
    assert "XLeRobot" in suite.ALL_ROBOTS
    assert ROBOT_CLASS_MAPPING["XLeRobot"] is WheeledRobot


def test_xlerobot_model_parts_and_arm_split():
    robot = WheeledRobot(robot_type="XLeRobot")
    robot.load_model()
    model = robot.robot_model

    assert model.arm_type == "bimanual"

    # Positional split convention: first half of arm_joints = "right", second = "left".
    arm_joints = model.arm_joints
    assert len(arm_joints) == 10
    assert arm_joints[:5] == [f"robot0_right_{n}" for n in ARM_JOINT_ORDER]
    assert arm_joints[5:] == [f"robot0_left_{n}" for n in ARM_JOINT_ORDER]

    # Substring-classified parts.
    assert model.head_joints == ["robot0_head_pan", "robot0_head_tilt"]
    assert len(model.base_joints) == 3
    assert all("mobile" in j for j in model.base_joints)
    assert model.torso_joints == []
    assert model.legs_joints == []

    # Arm-unique gripper namespaces (XLeRobotGripper).
    assert robot.gripper["right"].joints == ["robot0_right_gripper"]
    assert robot.gripper["left"].joints == ["robot0_left_gripper"]
    assert model.eef_name == {"right": "robot0_right_gripper", "left": "robot0_left_gripper"}

    # 17 real-motor channels: 10 arm + 2 head + 2 grippers (base velocities come
    # from the 3 virtual planar joints).
    assert len(model.arm_actuators) == 10
    assert len(model.head_actuators) == 2
    assert len(model.base_actuators) == 3


def test_xlerobot_frame_tree_matches_mjcf():
    """XLEROBOT_FRAME_TREE (used as the Tier-1 mount registry) must match the MJCF."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(XLEROBOT_XML))
    for body_name, spec in XLEROBOT_FRAME_TREE.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert bid >= 0, f"body {body_name} missing from robot.xml"
        assert np.allclose(m.body_pos[bid], spec["pos"], atol=1e-9), (
            f"{body_name}: MJCF pos {m.body_pos[bid]} != registry {spec['pos']}"
        )
        assert np.allclose(m.body_quat[bid], spec["quat"], atol=1e-9), (
            f"{body_name}: MJCF quat {m.body_quat[bid]} != registry {spec['quat']}"
        )


def test_xlerobot_init_qpos_within_joint_ranges():
    """Out-of-range init fires MuJoCo's limit constraint at reset and pollutes
    tau_ext (the shoulder_lift lesson) — guard every joint of the platform model."""
    root = ET.parse(XLEROBOT_XML).getroot()
    joint_order = (
        [f"right_{n}" for n in ARM_JOINT_ORDER]
        + [f"left_{n}" for n in ARM_JOINT_ORDER]
        + ["head_pan", "head_tilt"]
    )
    ranges = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in joint_order and joint.get("range"):
            ranges[name] = np.fromstring(joint.get("range"), sep=" ")
    assert set(ranges) == set(joint_order)

    init_qpos = XLeRobot(idn=0).init_qpos
    assert len(init_qpos) == len(joint_order)
    margin = 0.005
    for name, q0 in zip(joint_order, init_qpos):
        lo, hi = ranges[name]
        assert lo + margin <= q0 <= hi - margin, (
            f"{name}: init_qpos {q0:+.4f} outside safe range "
            f"[{lo + margin:+.4f}, {hi - margin:+.4f}]"
        )


def _quat_equal(q1, q2, atol=1e-6):
    """Quaternion equality up to sign (q and -q are the same rotation)."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return abs(float(np.dot(q1, q2))) > 1.0 - atol


def _farr(el, attr, default=None):
    v = el.get(attr, default)
    return None if v is None else np.fromstring(v, sep=" ")


CHAIN_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
CHAIN_SITES = ("gripperframe", "grip_site", "ee", "wrist_ft_site")


def test_xlerobot_arm_chains_match_soarm101_source():
    """Drift guard: both hand-transcribed arm chains in the XLeRobot MJCF must match
    soarm_with_sensor.xml numerically (a silent digit slip in an inertial or body pose
    would corrupt that arm's ground-truth forces). Root body pos/quat (the mount) is
    exempt; everything below it must be identical up to the right_/left_ prefix."""
    src_root = ET.parse(
        ASSET_ROOT / "robots" / "SOARM101" / "SO101" / "soarm_with_sensor.xml"
    ).getroot()
    dst_root = ET.parse(XLEROBOT_XML).getroot()

    def body_by_name(root, name):
        for b in root.iter("body"):
            if b.get("name") == name:
                return b
        raise AssertionError(f"body {name} not found")

    for prefix in ("right_", "left_"):
        for i, src_name in enumerate(CHAIN_BODIES):
            dst_name = f"{prefix}arm_base" if src_name == "base" else f"{prefix}{src_name}"
            src = body_by_name(src_root, src_name)
            dst = body_by_name(dst_root, dst_name)
            ctx = f"{dst_name} (from {src_name})"

            if i > 0:  # mount pose of the root body is the (intentionally different) transform
                assert np.allclose(_farr(src, "pos"), _farr(dst, "pos"), atol=1e-9), f"{ctx}: body pos"
                assert _quat_equal(
                    _farr(src, "quat", "1 0 0 0"), _farr(dst, "quat", "1 0 0 0")
                ), f"{ctx}: body quat"

            # Inertial must be bit-comparable (drives gravity/inertia GT terms).
            si, di = src.find("inertial"), dst.find("inertial")
            assert np.isclose(float(si.get("mass")), float(di.get("mass"))), f"{ctx}: mass"
            assert np.allclose(_farr(si, "pos"), _farr(di, "pos"), atol=1e-12), f"{ctx}: inertial pos"
            assert np.allclose(
                _farr(si, "fullinertia"), _farr(di, "fullinertia"), atol=1e-12
            ), f"{ctx}: fullinertia"

            # Joint (skip the root body, which has none).
            sj, dj = src.find("joint"), dst.find("joint")
            if sj is not None:
                assert dj is not None, f"{ctx}: joint missing"
                assert dj.get("name") == f"{prefix}{sj.get('name')}", f"{ctx}: joint name"
                assert np.allclose(_farr(sj, "axis"), _farr(dj, "axis")), f"{ctx}: joint axis"
                assert np.allclose(_farr(sj, "range"), _farr(dj, "range"), atol=1e-12), f"{ctx}: joint range"
                assert sj.get("class") == dj.get("class"), f"{ctx}: joint class"

            # Mesh geoms: multiset of (mesh, class, pos, canonical quat sign info dropped).
            def mesh_geoms(body):
                out = []
                for g in body.findall("geom"):
                    if g.get("mesh"):
                        q = _farr(g, "quat", "1 0 0 0")
                        q = q / np.linalg.norm(q)
                        if q[np.argmax(np.abs(q))] < 0:
                            q = -q
                        out.append(
                            (g.get("mesh"), g.get("class"), tuple(np.round(_farr(g, "pos"), 9)),
                             tuple(np.round(q, 6)))
                        )
                return sorted(out)

            assert mesh_geoms(src) == mesh_geoms(dst), f"{ctx}: mesh geoms differ"

            # Named finger-pad boxes (grasp-critical collision).
            def box_geoms(body, strip=""):
                out = {}
                for g in body.findall("geom"):
                    name = g.get("name") or ""
                    if "static_finger" in name:
                        key = name[len(strip):] if name.startswith(strip) else name
                        out[key] = (
                            tuple(np.round(_farr(g, "pos"), 9)),
                            tuple(np.round(_farr(g, "size"), 9)),
                        )
                return out

            assert box_geoms(src) == box_geoms(dst, strip=prefix), f"{ctx}: finger pads differ"

        # EE-frame sites (TCP definition used by wrenches/OSC).
        src_grip = body_by_name(src_root, "gripper")
        dst_grip = body_by_name(dst_root, f"{prefix}gripper")
        src_sites = {s.get("name"): s for s in src_grip.findall("site")}
        dst_sites = {s.get("name"): s for s in dst_grip.findall("site")}
        for site in CHAIN_SITES:
            ss, ds = src_sites[site], dst_sites[f"{prefix}{site}"]
            assert np.allclose(_farr(ss, "pos"), _farr(ds, "pos"), atol=1e-9), f"{prefix}{site}: pos"
            assert _quat_equal(
                _farr(ss, "quat", "1 0 0 0"), _farr(ds, "quat", "1 0 0 0")
            ), f"{prefix}{site}: quat"


def test_xlerobot_env_action_dim_sensors_and_steps():
    import mujoco

    env = suite.make(
        env_name="Lift",
        robots=["XLeRobot"],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        initialization_noise=None,
    )
    try:
        env.reset()
        low, high = env.action_spec
        # [right arm 5 + grip 1, left arm 5 + grip 1, head 2, base 3] = 17
        assert low.shape[0] == 17, f"action dim {low.shape[0]} != 17"

        model = env.sim.model._model
        for sensor_name in (
            "robot0_right_wrist_ft_force",
            "robot0_right_wrist_ft_torque",
            "robot0_left_wrist_ft_force",
            "robot0_left_wrist_ft_torque",
        ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0, (
                f"sensor {sensor_name} missing"
            )

        # No joint starts outside its range (limit-constraint guard, live model).
        for name in env.robots[0].robot_model.joints:
            j = model.joint(name)
            if j.limited:
                q = env.sim.data.qpos[model.jnt_qposadr[j.id]]
                assert j.range[0] <= q <= j.range[1], f"{name} starts at {q} outside {j.range}"

        for _ in range(5):
            obs, reward, done, info = env.step(np.zeros(low.shape[0]))
        assert np.isfinite(np.asarray(env.sim.data.qpos)).all()
        assert np.isfinite(np.asarray(env.sim.data.qvel)).all()
    finally:
        env.close()


def test_pnp_xlerobot_base_resists_external_force():
    import mujoco

    env = suite.make(
        env_name="cupPnP_task1",
        robots=["XLeRobot"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
        control_freq=20,
    )
    try:
        env.reset()
        assert type(env.robots[0].robot_model.base).__name__ == "LockedNullMobileBase"
        assert env.action_dim == 17

        model = env.sim.model._model
        data = env.sim.data._data
        qpos_adr = []
        qvel_adr = []
        for name in env.robots[0].robot_model.base_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr.append(model.jnt_qposadr[joint_id])
            qvel_adr.append(model.jnt_dofadr[joint_id])

        support_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support"
        )
        qpos_before = data.qpos[qpos_adr].copy()
        xpos_before = data.xpos[support_id].copy()
        xmat_before = data.xmat[support_id].copy()

        # Much larger than normal arm/table contact: the cart should still stay parked.
        data.xfrc_applied[support_id] = [1000.0, -800.0, 0.0, 0.0, 0.0, 500.0]
        action = env.robots[0].create_action_vector(
            {"base": np.array([1.0, -1.0, 1.0])}
        )
        for _ in range(50):
            env.step(action)

        assert np.allclose(data.qpos[qpos_adr], qpos_before, atol=1e-5)
        assert np.allclose(data.qvel[qvel_adr], 0.0, atol=1e-8)
        assert np.allclose(data.xpos[support_id], xpos_before, atol=1e-5)
        assert np.allclose(data.xmat[support_id], xmat_before, atol=1e-5)
    finally:
        env.close()

def test_task5_xlerobot_cup_starts_on_blind_holder_floor():
    from robosuite.environments.manipulation.cup_pnp_task5 import cupPnP_task5
    from robosuite.environments.manipulation.soarm101_lift import (
        XLEROBOT_CUP_HOLDER_CENTER,
        XLEROBOT_CUP_HOLDER_HALF_SIZE,
        XLEROBOT_CUP_HOLDER_SUPPORT_Z,
    )

    holder = ET.parse(XLEROBOT_XML).getroot().find(
        ".//geom[@name='cup_foam_place_bottom']"
    )
    holder_pos = _farr(holder, "pos")
    holder_size = _farr(holder, "size")
    assert np.allclose(XLEROBOT_CUP_HOLDER_CENTER[:2], holder_pos[:2])
    assert np.allclose(XLEROBOT_CUP_HOLDER_HALF_SIZE, holder_size[:2])
    assert np.isclose(XLEROBOT_CUP_HOLDER_SUPPORT_Z, holder_pos[2] + holder_size[2])

    root = ET.parse(XLEROBOT_XML).getroot()
    foam_geoms = [
        geom
        for geom in root.findall(".//geom")
        if geom.get("name", "").startswith("cup_foam_")
        and geom.get("name") != "cup_foam_place_bottom"
    ]
    assert foam_geoms
    for geom in foam_geoms:
        pos = _farr(geom, "pos")
        size = _farr(geom, "size")
        assert np.isclose(pos[2] + size[2], 0.775)

    task = cupPnP_task5.__new__(cupPnP_task5)
    task.robot_is_self_supporting = True
    task.robot_world_base_pos = np.array([-0.64, 0.02, 0.0])
    task._layout_rot_z = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    task.cup_size = np.array([0.03, 0.05])

    start = task._cart_start_pos()
    expected_xy = (
        task.robot_world_base_pos + task._layout_rot_z @ XLEROBOT_CUP_HOLDER_CENTER
    )[:2]
    assert np.allclose(start[:2], expected_xy)
    assert np.isclose(start[2], XLEROBOT_CUP_HOLDER_SUPPORT_Z + 0.05 + 0.002)


def test_task3_xlerobot_success_uses_blind_holder_not_legacy_cart_top():
    from robosuite.environments.manipulation.cup_pnp_task3 import cupPnP_task3
    from robosuite.environments.manipulation.soarm101_lift import (
        XLEROBOT_CUP_HOLDER_CENTER,
        XLEROBOT_CUP_HOLDER_SUPPORT_Z,
    )

    task = cupPnP_task3.__new__(cupPnP_task3)
    task.robot_is_self_supporting = True
    task.robot_world_base_pos = np.array([-0.606, 0.02, 0.0])
    task.robot_support_table_yaw = np.deg2rad(30.0)
    task.cart_top = np.array([-0.585, 0.0, 0.770])
    task.cup_size = np.array([0.030, 0.050])
    task.holder_center_tolerance_m = 0.012

    yaw = task.robot_support_table_yaw
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    holder_center = task.robot_world_base_pos + rot_z @ XLEROBOT_CUP_HOLDER_CENTER
    cup_pos = holder_center.copy()
    cup_pos[2] = XLEROBOT_CUP_HOLDER_SUPPORT_Z + task.cup_size[1]
    task._cup_pos = lambda: cup_pos

    target = task._cart_target_pos()
    assert np.allclose(target[:2], holder_center[:2])
    assert np.isclose(target[2], cup_pos[2] + 0.002)
    assert task._cup_on_cart()
    assert type(task._cup_on_cart()) is bool

    # A visually inserted cup can settle slightly off-center in the collision
    # geometry and should still count as being in this specific holder.
    cup_pos[:] = holder_center + rot_z @ np.array([0.009, 0.0, 0.0])
    cup_pos[2] = XLEROBOT_CUP_HOLDER_SUPPORT_Z + task.cup_size[1]
    assert task._cup_on_cart()

    cup_pos[:] = holder_center + rot_z @ np.array([0.0115, 0.0, 0.0])
    cup_pos[2] = XLEROBOT_CUP_HOLDER_SUPPORT_Z + task.cup_size[1]
    assert task._cup_on_cart()

    # The obsolete task-3 gate used z=0.770. A cup on the holder rim at that
    # height is not inserted into the blind slot and must not count as success.
    cup_pos[2] = task.cart_top[2] + task.cup_size[1]
    assert not task._cup_on_cart()

    # Nor should an upright cup elsewhere on the cart count as being in the slot.
    cup_pos[:] = holder_center + rot_z @ np.array([0.020, 0.0, 0.0])
    cup_pos[2] = XLEROBOT_CUP_HOLDER_SUPPORT_Z + task.cup_size[1]
    assert not task._cup_on_cart()


def test_task3_xlerobot_env_accepts_cup_on_blind_holder_floor():
    env = suite.make(
        env_name="cupPnP_task3",
        robots=["XLeRobot"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
    )
    try:
        env.reset()
        assert env.robot_is_self_supporting

        placed_pos = env._cart_target_pos()
        placed_pos[2] -= 0.002
        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([placed_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6))
        env.sim.forward()
        assert env._cup_on_cart()

        placed_pos[2] = env.cart_top[2] + float(env.cup_size[1])
        env.sim.data.set_joint_qpos(
            env.cube.joints[0],
            np.concatenate([placed_pos, np.array([1.0, 0.0, 0.0, 0.0])]),
        )
        env.sim.forward()
        assert not env._cup_on_cart()
    finally:
        env.close()


def test_task5_xlerobot_faces_table_with_one_cm_white_tray_gap():
    import mujoco

    env = suite.make(
        env_name="cupPnP_task5",
        robots=["XLeRobot"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=False,
    )
    try:
        env.reset()
        assert np.isclose(env.layout_yaw_rad, 0.0)

        model = env.sim.model._model
        data = env.sim.data._data
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "robot0_cart_body_vis"
        )
        mesh_id = model.geom_dataid[geom_id]
        first = model.mesh_vertadr[mesh_id]
        last = first + model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[first:last]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]

        white_tray_front_x = float(np.max(world_vertices[:, 0]))
        table_near_x = float(env.table_offset[0] - env.table_full_size[0] / 2.0)
        gap = table_near_x - white_tray_front_x
        assert np.isclose(gap, 0.01, atol=0.003)

        robot_to_table = np.asarray(env.table_offset[:2]) - np.asarray(
            env.robot_world_base_pos[:2]
        )
        robot_forward = np.array(
            [np.cos(env.layout_yaw_rad), np.sin(env.layout_yaw_rad)]
        )
        assert np.dot(robot_forward, robot_to_table) > 0.0
    finally:
        env.close()

def test_task1_xlerobot_reset_uses_calibrated_head_pose(monkeypatch):
    from unittest.mock import MagicMock
    from robosuite.environments.manipulation.cup_pnp_task1 import cupPnP_task1
    from robosuite.environments.manipulation.soarm101_lift import SOARM101Lift

    monkeypatch.setattr(SOARM101Lift, "_reset_internal", lambda self: None)
    task = cupPnP_task1.__new__(cupPnP_task1)
    task.robot_is_self_supporting = True
    task.head_start_pan_deg = 8.0
    task.head_start_tilt_deg = 32.0
    task.sim = MagicMock()
    task._reset_cup_to_table_start = lambda: None
    task.cup_color_randomization = 0.0
    task._randomize_cup_visual_color = lambda amount: None
    task._update_success_target_visual = lambda success: None

    task._reset_internal()
    qpos = {
        call.args[0]: call.args[1]
        for call in task.sim.data.set_joint_qpos.call_args_list
    }
    assert np.isclose(np.degrees(qpos["robot0_head_pan"]), 8.0)
    assert np.isclose(np.degrees(qpos["robot0_head_tilt"]), 32.0)
    task.sim.forward.assert_called_once_with()
