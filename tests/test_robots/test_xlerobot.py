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
