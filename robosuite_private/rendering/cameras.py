"""
get_camera_frames(env, camera_names, width, height) -> dict[str, np.ndarray]

Returns RGB uint8 H×W×3 frames for each requested camera, rendered via the
environment's offscreen renderer.  The env must have been created with
has_offscreen_renderer=True.
"""

from __future__ import annotations

import numpy as np


def get_camera_frames(
    env,
    camera_names: list[str] | None = None,
    width:  int = 640,
    height: int = 480,
) -> dict[str, np.ndarray]:
    """
    Render camera frames from the current simulation state.

    Args:
        env:          robosuite environment (must have offscreen renderer).
        camera_names: list of camera names to render.  Defaults to all
                      cameras in env.camera_names.
        width:        frame width  [pixels].
        height:       frame height [pixels].

    Returns:
        dict mapping camera_name -> RGB uint8 ndarray of shape (H, W, 3).
    """
    if not env.has_offscreen_renderer:
        raise RuntimeError(
            "get_camera_frames requires has_offscreen_renderer=True"
        )

    if camera_names is None:
        camera_names = list(env.camera_names) if env.camera_names else []

    # Match robosuite's image convention (robot_env uses the same pattern).
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

    frames: dict[str, np.ndarray] = {}
    for name in camera_names:
        raw = env.sim.render(
            camera_name=name,
            width=width,
            height=height,
            depth=False,
        )
        frames[name] = raw[::convention].astype(np.uint8)

    return frames


def get_camera_rgbd_frames(
    env,
    camera_names: list[str] | None = None,
    width: int = 640,
    height: int = 480,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Render RGB plus metric uint16 depth in millimetres."""
    if not env.has_offscreen_renderer:
        raise RuntimeError(
            "get_camera_rgbd_frames requires has_offscreen_renderer=True"
        )

    if camera_names is None:
        camera_names = list(env.camera_names) if env.camera_names else []

    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in camera_names:
        rgb, normalized_depth = env.sim.render(
            camera_name=name,
            width=width,
            height=height,
            depth=True,
        )
        extent = env.sim.model.stat.extent
        far = env.sim.model.vis.map.zfar * extent
        near = env.sim.model.vis.map.znear * extent
        depth_m = near / (1.0 - normalized_depth * (1.0 - near / far))
        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        frames[name] = (
            rgb[::convention].astype(np.uint8),
            depth_mm[::convention],
        )

    return frames
