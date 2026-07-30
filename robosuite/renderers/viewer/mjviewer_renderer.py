import copy

import mujoco
from mujoco import viewer

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 1],
    "distance": 2,
    "azimuth": 180,
    "elevation": -20,
}


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None, geom_alpha_overrides=None):
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
        self._viewer_model = None
        self._viewer_data = None

    def _initialize_viewer_sim(self):
        source_model = self.env.sim.model._model
        source_data = self.env.sim.data._data
        if not self.geom_alpha_overrides:
            self._viewer_model = source_model
            self._viewer_data = source_data
            return

        # Keep display-only opacity out of the simulation model used by
        # offscreen dataset cameras.
        self._viewer_model = copy.deepcopy(source_model)
        self._viewer_data = mujoco.MjData(self._viewer_model)
        mujoco.mj_copyData(self._viewer_data, self._viewer_model, source_data)
        self._sync_viewer_visuals()

    def _sync_viewer_visuals(self):
        source_model = self.env.sim.model._model
        self._viewer_model.geom_rgba[:] = source_model.geom_rgba
        self._viewer_model.mat_rgba[:] = source_model.mat_rgba
        self._viewer_model.site_rgba[:] = source_model.site_rgba

        modified_materials = set()
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
            if alpha is None:
                continue

            material_id = int(self._viewer_model.geom_matid[geom_id])
            if material_id >= 0:
                if material_id not in modified_materials:
                    self._viewer_model.mat_rgba[material_id, 3] *= alpha
                    modified_materials.add(material_id)
            else:
                self._viewer_model.geom_rgba[geom_id, 3] *= alpha

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

        if self._viewer_data is not self.env.sim.data._data:
            with self.viewer.lock():
                self._sync_viewer_visuals()
                mujoco.mj_copyData(
                    self._viewer_data,
                    self._viewer_model,
                    self.env.sim.data._data,
                )
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

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
