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
        return np.array([0.0, 0.65, 0.0, 1.5, 0.0])

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