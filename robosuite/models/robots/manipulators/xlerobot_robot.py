import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

# Tier-1 frame tree (Project_definition.md §3): mount transforms of the arm bases,
# head pan link, and head tilt link in the robot base frame B (+x forward, +y left,
# z up, origin on the floor under the cart center). Values are initial estimates
# extracted from upstream Vector-Wangel/XLeRobot simulation/mujoco/xlerobot.xml
# @ 3d14695e (world-frame poses remapped by Rz(-90 deg); upstream forward = +y).
# Arms are mounted flat -> pure yaw, assumed 0 (both face forward). MUST be
# verified against our physical build (doc §3.3 Validations A/C) before platform
# force data is trusted. test_xlerobot.py guards these against the MJCF.
XLEROBOT_FRAME_TREE = {
    "right_arm_base": {"pos": (0.1352, -0.15, 0.8215), "quat": (1.0, 0.0, 0.0, 0.0)},
    "left_arm_base": {"pos": (0.1352, 0.15, 0.8215), "quat": (1.0, 0.0, 0.0, 0.0)},
    # head_tilt_link pos is relative to head_pan_link
    "head_pan_link": {"pos": (-0.125, 0.0, 0.945), "quat": (1.0, 0.0, 0.0, 0.0)},
    "head_tilt_link": {"pos": (0.05, 0.0, 0.18), "quat": (1.0, 0.0, 0.0, 0.0)},
    # GY-91 at the 2nd tray layer's geometric centre (z estimated — measure on
    # the build). E-FC compensation transfers its readings to each arm base.
    "imu": {"pos": (0.0, 0.0, 0.42), "quat": (1.0, 0.0, 0.0, 0.0)},
}


# Registered as WheeledRobot in robosuite.robots.ROBOT_CLASS_MAPPING (avoids a
# models -> robots -> models circular import; same pattern as SOARM101).
class XLeRobot(ManipulatorModel):
    """
    XLeRobot: dual SO-101 arms + 2-DoF head on an omni-wheel cart.

    17 real motors: 2 x (5 arm + 1 gripper) + 2 head + 3 wheels. In sim the wheels
    are replaced by the 3 virtual planar base joints of NullMobileBase (forward/
    side/yaw velocity), which matches the real robot's base command interface
    (body-frame x.vel / y.vel / theta.vel); kiwi-drive wheel mixing stays outside
    the sim, exactly as it stays on the real host. Default composite controller
    (default_xlerobot.json) yields a 17-dim action, composite order
    [right arm 5, left arm 5, head 2, base 3, right grip 1, left grip 1]
    (build actions with robot.create_action_vector to stay layout-proof).

    Args:
        idn (int or str): Number or some other unique identification string for
            this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/XLeRobot/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "NullMobileBase"

    @property
    def default_gripper(self):
        return {"right": "XLeRobotGripper", "left": "XLeRobotGripper"}

    @property
    def default_controller_config(self):
        return {
            "right": "joint_position",
            "left": "joint_position",
            "head": "joint_position",
        }

    @property
    def init_qpos(self):
        """
        Document-order qpos for all robot-XML joints: [right arm 5, left arm 5, head 2].

        Both arms use the SOARM101 scene-match pose (shoulder_lift clamped inside the
        model's +/-1.7453 range -- see soarm101_robot.py); head level, centered.
        """
        arm = [0.0875, -1.7353, 1.5443, 1.3579, 1.6809]
        return np.array(arm + arm + [0.0, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.30 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.2))

    @property
    def _horizontal_radius(self):
        return 0.30

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_gripper", "left": "left_gripper"}
