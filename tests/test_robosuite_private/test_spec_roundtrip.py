"""
Spec round-trip: export_robot_spec() joint limits must match what the
MuJoCo model actually loads, within 1e-4 rad.
"""

import pytest
import mujoco
from robosuite.utils.mjcf_utils import xml_path_completion

from robosuite_private.robot_spec import export_robot_spec, validate_against_mjcf


@pytest.fixture(scope="module")
def mj_model():
    xml = xml_path_completion("robots/SOARM101/SO101/soarm_with_sensor.xml")
    return mujoco.MjModel.from_xml_path(xml)


def test_export_has_required_keys():
    spec = export_robot_spec()
    required = {
        "version", "robot", "joint_names", "gripper_joint",
        "all_joint_names", "joint_limits", "joint_damping",
        "joint_frictionloss", "joint_armature", "kt_default",
        "servo_force_range",
    }
    assert required.issubset(spec.keys())


def test_export_joint_limits_match_mjcf(mj_model):
    """Joint limits in canonical.py must match soarm_with_sensor.xml within 1e-4 rad."""
    spec = export_robot_spec()
    TOL  = 1e-4

    for jname, (lo, hi) in spec["joint_limits"].items():
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue  # joint not present in standalone XML (may be prefixed in robosuite)
        actual_lo = float(mj_model.jnt_range[jid, 0])
        actual_hi = float(mj_model.jnt_range[jid, 1])
        assert abs(actual_lo - lo) < TOL, (
            f"{jname} lower limit: canonical={lo:.6f} mjcf={actual_lo:.6f}"
        )
        assert abs(actual_hi - hi) < TOL, (
            f"{jname} upper limit: canonical={hi:.6f} mjcf={actual_hi:.6f}"
        )


def test_validate_against_mjcf_passes(mj_model):
    """validate_against_mjcf() must not raise for the production XML."""
    validate_against_mjcf(mj_model)  # raises AssertionError on mismatch
