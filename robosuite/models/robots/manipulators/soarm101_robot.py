import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SOARM101(ManipulatorModel):
    """
    SOARM101 single-arm robot.

    The SO101 XML includes the built-in wrist/gripper mechanism, so this model
    treats the arm as a single fixed-base manipulator and keeps the standard
    robosuite gripper interface as a mount adapter.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/SOARM101/SO101/soarm_with_sensor.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMinimalMount"

    @property
    def default_gripper(self):
        return {"right": "SO101Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "osc_position"}

    @property
    def init_qpos(self):
        # Episode-1 frame-0 pose from hpi_boxes_standard (exact):
        #   pan 5.01, shoulder_lift -104.13, elbow_flex 96.48, wrist_flex 77.80, wrist_roll 96.31 (deg).
        # Two joints deviate from the real pose:
        #   - elbow_flex unfolded ~8 deg (96.5 -> 88.5) so the folded gripper clears the table
        #     (the exact elbow dips the gripper ~18 mm into the table -- a sim height offset;
        #     the real pose is collision-free).
        #   - shoulder_lift clamped -1.8174 -> -1.7353: the real calibrated range exceeds the
        #     model's +/-1.7453 rad limit, and an out-of-range init fires MuJoCo's joint-limit
        #     constraint at reset (~4-7 N.m phantom torque that breaks the tau_ext_residual
        #     free-space identity and pollutes episode-start force data). 0.01 rad margin.
        # [pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] (rad). Reset is deterministic.
        return np.array([0.0875, -1.7353, 1.5443, 1.3579, 1.6809])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.55))

    @property
    def _horizontal_radius(self):
        return 0.35

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        return {"right": "gripper"}