"""
Gripper adapter for the SOARM101 end-effector.
"""

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SO101Gripper(GripperModel):
    """
    Custom gripper wrapper for SOARM101.

    This model provides the standard robosuite gripper sites and force/torque
    sensor frame while keeping the SO101 jaw mechanism controlled in the arm XML.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/so101_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None