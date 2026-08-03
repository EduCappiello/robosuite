from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class Room128Arena(Arena):
    """Room 128 — the classroom the real XLeRobot coffee dataset was recorded in.

    Geometry is a 1:1 copy of ``environments/Room_128/classroom128_v2.usd`` (Z-up,
    metres, all Cube prims), re-centred on the world origin so the coffee-station
    furniture the tasks already build around (0, 0) lands inside the room. Inner
    floor 3.665 x 4.935 m, walls 2.8 m.

    Unlike robosuite's stock arenas, the walls and furniture are **collidable** —
    this arena exists so the cart can be driven around the room (the real dataset's
    ``t4_navigate`` task), which needs obstacles that actually stop it.

    Regenerate with ``scripts/usd_to_room128_arena.py`` if the USD changes; do not
    hand-edit the XML.
    """

    def __init__(self):
        super().__init__(xml_path_completion("arenas/room128_arena.xml"))
