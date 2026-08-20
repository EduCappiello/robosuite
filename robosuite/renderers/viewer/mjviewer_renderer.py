import copy

import mujoco
import numpy as np
from mujoco import viewer

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 1],
    "distance": 2,
    "azimuth": 180,
    "elevation": -20,
}


class MjviewerRenderer:
    def __init__(
        self,
        env,
        camera_id=None,
        cam_config=None,
        geom_alpha_overrides=None,
        geom_rgb_overrides=None,
        body_alpha_overrides=None,
        body_alpha_exclude_prefixes=None,
        gripper_alignment_site_names=None,
        gripper_alignment_center_geom_pairs=None,
        gripper_alignment_line_length=0.12,
        gripper_alignment_line_width=3.0,
        gripper_alignment_line_rgba=(0.0, 1.0, 0.0, 0.85),
    ):
        if cam_config is None:
            cam_config = DEFAULT_FREE_CAM
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config
        self.keypress_callback = None
        self.geom_alpha_overrides = {
            str(prefix): float(alpha)
            for prefix, alpha in (geom_alpha_overrides or {}).items()
        }
        for prefix, alpha in self.geom_alpha_overrides.items():
            if not prefix:
                raise ValueError("Viewer geom alpha prefix cannot be empty")
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Viewer alpha for {prefix!r} must be between 0 and 1")
        self.geom_rgb_overrides = {
            str(prefix): tuple(float(channel) for channel in rgb)
            for prefix, rgb in (geom_rgb_overrides or {}).items()
        }
        for prefix, rgb in self.geom_rgb_overrides.items():
            if not prefix:
                raise ValueError("Viewer geom RGB prefix cannot be empty")
            if len(rgb) != 3 or any(channel < 0.0 or channel > 1.0 for channel in rgb):
                raise ValueError(
                    f"Viewer RGB for {prefix!r} must contain three values between 0 and 1"
                )

        self.body_alpha_overrides = {
            str(prefix): float(alpha)
            for prefix, alpha in (body_alpha_overrides or {}).items()
        }
        for prefix, alpha in self.body_alpha_overrides.items():
            if not prefix:
                raise ValueError("Viewer body alpha prefix cannot be empty")
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Viewer alpha for body {prefix!r} must be between 0 and 1")
        self.body_alpha_exclude_prefixes = tuple(
            str(prefix) for prefix in (body_alpha_exclude_prefixes or ())
        )
        if any(not prefix for prefix in self.body_alpha_exclude_prefixes):
            raise ValueError("Viewer body alpha exclusion prefix cannot be empty")
        self.gripper_alignment_site_names = tuple(
            str(name) for name in (gripper_alignment_site_names or ())
        )
        if any(not name for name in self.gripper_alignment_site_names):
            raise ValueError("Viewer gripper alignment site name cannot be empty")
        self.gripper_alignment_center_geom_pairs = {
            str(site_name): tuple(str(geom_name) for geom_name in geom_names)
            for site_name, geom_names in (gripper_alignment_center_geom_pairs or {}).items()
        }
        for site_name, geom_names in self.gripper_alignment_center_geom_pairs.items():
            if not site_name or len(geom_names) != 2 or any(not name for name in geom_names):
                raise ValueError(
                    "Viewer gripper alignment center must map a site to two geom names"
                )
        self.gripper_alignment_line_length = float(gripper_alignment_line_length)
        if self.gripper_alignment_line_length <= 0.0:
            raise ValueError("Viewer gripper alignment line length must be positive")
        self.gripper_alignment_line_width = float(gripper_alignment_line_width)
        if self.gripper_alignment_line_width <= 0.0:
            raise ValueError("Viewer gripper alignment line width must be positive")
        self.gripper_alignment_line_rgba = np.asarray(
            gripper_alignment_line_rgba, dtype=np.float32
        )
        if self.gripper_alignment_line_rgba.shape != (4,) or np.any(
            (self.gripper_alignment_line_rgba < 0.0)
            | (self.gripper_alignment_line_rgba > 1.0)
        ):
            raise ValueError(
                "Viewer gripper alignment RGBA must contain four values between 0 and 1"
            )

        self._viewer_model = None
        self._viewer_data = None
        self._gripper_alignment_geom_start = None

    def _initialize_viewer_sim(self):
        source_model = self.env.sim.model._model
        source_data = self.env.sim.data._data
        if (
            not self.geom_alpha_overrides
            and not self.geom_rgb_overrides
            and not self.body_alpha_overrides
        ):
            self._viewer_model = source_model
            self._viewer_data = source_data
            return

        # Keep display-only opacity out of the simulation model used by
        # offscreen dataset cameras.
        self._viewer_model = copy.deepcopy(source_model)
        self._viewer_data = mujoco.MjData(self._viewer_model)
        mujoco.mj_copyData(self._viewer_data, self._viewer_model, source_data)
        self._sync_viewer_visuals()

    def _body_alpha_for_geom(self, geom_id):
        body_id = int(self._viewer_model.geom_bodyid[geom_id])
        while body_id > 0:
            body_name = self._viewer_model.body(body_id).name or ""
            if body_name.startswith(self.body_alpha_exclude_prefixes):
                return None
            for prefix, alpha in self.body_alpha_overrides.items():
                if body_name.startswith(prefix):
                    return alpha
            body_id = int(self._viewer_model.body_parentid[body_id])
        return None

    def _sync_viewer_visuals(self):
        source_model = self.env.sim.model._model
        self._viewer_model.geom_rgba[:] = source_model.geom_rgba
        self._viewer_model.geom_matid[:] = source_model.geom_matid
        self._viewer_model.mat_rgba[:] = source_model.mat_rgba
        self._viewer_model.site_rgba[:] = source_model.site_rgba

        alpha_modified_materials = set()
        rgb_modified_materials = set()
        for geom_id in range(self._viewer_model.ngeom):
            name = self._viewer_model.geom(geom_id).name or ""
            alpha = next(
                (
                    value
                    for prefix, value in self.geom_alpha_overrides.items()
                    if name.startswith(prefix)
                ),
                None,
            )

            body_alpha = self._body_alpha_for_geom(geom_id)
            if alpha is None:
                alpha = body_alpha

            rgb = next(
                (
                    value
                    for prefix, value in self.geom_rgb_overrides.items()
                    if name.startswith(prefix)
                ),
                None,
            )
            if alpha is None and rgb is None:
                continue

            material_id = int(self._viewer_model.geom_matid[geom_id])
            if material_id >= 0:
                if alpha is not None and body_alpha is not None:
                    # A body override may select an arm geom while excluding an
                    # opaque gripper geom that shares the same material.
                    self._viewer_model.geom_rgba[geom_id] = source_model.mat_rgba[material_id]
                    self._viewer_model.geom_rgba[geom_id, 3] *= alpha
                    self._viewer_model.geom_matid[geom_id] = -1
                elif alpha is not None and material_id not in alpha_modified_materials:
                    self._viewer_model.mat_rgba[material_id, 3] *= alpha
                    alpha_modified_materials.add(material_id)
                if rgb is not None and material_id not in rgb_modified_materials:
                    self._viewer_model.mat_rgba[material_id, :3] = rgb
                    rgb_modified_materials.add(material_id)
            else:
                if alpha is not None:
                    self._viewer_model.geom_rgba[geom_id, 3] *= alpha
                if rgb is not None:
                    self._viewer_model.geom_rgba[geom_id, :3] = rgb

    def _sync_gripper_alignment_lines(self):
        if not self.gripper_alignment_site_names:
            return

        scene = self.viewer.user_scn
        line_count = len(self.gripper_alignment_site_names)
        if (
            self._gripper_alignment_geom_start is None
            or scene.ngeom < self._gripper_alignment_geom_start + line_count
        ):
            self._gripper_alignment_geom_start = scene.ngeom
            if scene.ngeom + line_count > len(scene.geoms):
                raise RuntimeError("MuJoCo viewer user scene has no room for alignment lines")
            scene.ngeom += line_count
        half_length = self.gripper_alignment_line_length / 2.0

        for line_index, site_name in enumerate(self.gripper_alignment_site_names):
            site_id = mujoco.mj_name2id(
                self._viewer_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise ValueError(f"Viewer gripper alignment site not found: {site_name!r}")

            center = np.asarray(self._viewer_data.site_xpos[site_id], dtype=np.float64)
            center_geom_names = self.gripper_alignment_center_geom_pairs.get(site_name)
            if center_geom_names is not None:
                center_geom_ids = [
                    mujoco.mj_name2id(
                        self._viewer_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                    )
                    for geom_name in center_geom_names
                ]
                missing_geom_names = [
                    name
                    for name, geom_id in zip(center_geom_names, center_geom_ids, strict=True)
                    if geom_id < 0
                ]
                if missing_geom_names:
                    raise ValueError(
                        "Viewer gripper alignment center geom not found: "
                        + ", ".join(repr(name) for name in missing_geom_names)
                    )
                center = np.mean(
                    self._viewer_data.geom_xpos[center_geom_ids], axis=0, dtype=np.float64
                )
            site_rotation = np.asarray(
                self._viewer_data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3)
            # The SO101 alignment guide runs vertically through the center of
            # the jaw opening when viewed from the gripper front.
            axis = site_rotation[:, 1]
            start = center - half_length * axis
            end = center + half_length * axis

            line = scene.geoms[self._gripper_alignment_geom_start + line_index]
            mujoco.mjv_initGeom(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3, dtype=np.float64),
                center,
                np.eye(3, dtype=np.float64).reshape(-1),
                self.gripper_alignment_line_rgba,
            )
            mujoco.mjv_connector(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                self.gripper_alignment_line_width,
                start,
                end,
            )

    def render(self):
        pass

    def set_camera(self, camera_id):
        self.camera_id = camera_id
        self._apply_camera()

    def _apply_camera(self):
        if self.viewer is None or self.camera_id is None:
            return
        if self.camera_id >= 0:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.camera_id
        else:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def update(self):
        if self.viewer is None:
            self._initialize_viewer_sim()
            self.viewer = viewer.launch_passive(
                self._viewer_model,
                self._viewer_data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.keypress_callback,
            )

            self.viewer.opt.geomgroup[0] = 1 if self.env.render_collision_mesh else 0
            self.viewer.opt.geomgroup[1] = 1 if self.env.render_visual_mesh else 0

            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]

            self._apply_camera()

        with self.viewer.lock():
            if self._viewer_data is not self.env.sim.data._data:
                self._sync_viewer_visuals()
                mujoco.mj_copyData(
                    self._viewer_data,
                    self._viewer_model,
                    self.env.sim.data._data,
                )
            self._sync_gripper_alignment_lines()
        self.viewer.sync()

    def reset(self):
        pass

    def close(self):

        self.sim = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._viewer_model = None
        self._viewer_data = None
        self._gripper_alignment_geom_start = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
