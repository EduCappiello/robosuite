"""Planar mobile-base interface whose pose is constrained to its reset pose."""

import numpy as np

from robosuite.models.bases.mobile_base_model import MobileBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class LockedNullMobileBase(MobileBaseModel):
    """Preserve base action/state channels while making the chassis immovable."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/locked_null_mobile_base.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0