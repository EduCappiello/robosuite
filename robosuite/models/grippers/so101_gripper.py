"""
Gripper adapter for the SOARM101 end-effector.
"""

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

import numpy as np


class SO101Gripper(GripperModel):
    """
    Custom gripper wrapper for SOARM101.

    This model provides the standard robosuite gripper sites and force/torque
    sensor frame while keeping the SO101 jaw mechanism controlled in the arm XML.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/so101_gripper.xml"), idn=idn)

    @property
    def naming_prefix(self):
        return f"robot{str(self.idn).split('_', 1)[0]}_"

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def dof(self):
        return 1

    @property
    def joints(self):
        return self.correct_naming(["gripper"])

    @property
    def actuators(self):
        return self.correct_naming(["gripper"])

    @property
    def _important_sites(self):
        return {
            "grip_site": "so101_grip_site",
            "grip_cylinder": "so101_grip_site_cylinder",
            "ee": "so101_ee",
            "ee_x": "so101_ee_x",
            "ee_y": "so101_ee_y",
            "ee_z": "so101_ee_z",
        }