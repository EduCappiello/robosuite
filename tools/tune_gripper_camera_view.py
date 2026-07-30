"""Interactive tuner for the SO101 gripper camera optical view.

Run from the robosuite repo root:
    conda run -n lerobot python tools/tune_gripper_camera_view.py

This tunes only the MuJoCo <camera name="gripper_cam"> element. It does not move
the white mount, black PCB, lens, or any collision geometry. Use it when the
physical camera placement is OK but the rendered camera view needs local optical
axis / FOV adjustment.
"""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import robosuite as suite
import robosuite_private  # noqa: F401 - registers private envs / robots


CAMERA_NAME = "robot0_gripper_cam"
DEFAULT_FOVY = 78.0
PREVIEW_W = 320
PREVIEW_H = 240


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


class CameraViewTuner:
    def __init__(self) -> None:
        self.env = suite.make(
            env_name="SOARM101Lift",
            robots="SOARM101",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            camera_names=[],
        )
        self.env.reset()
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data

        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if self.camera_id < 0:
            raise RuntimeError(f"Could not find camera {CAMERA_NAME!r}. Check so101_gripper.xml.")

        self._make_tuning_scene_visible()

        self.cam_pos = np.array(self.model.cam_pos[self.camera_id], dtype=float)
        self.base_quat = np.array(self.model.cam_quat[self.camera_id], dtype=float)
        self.delta_rpy_deg = np.array([0.0, 0.0, 0.0], dtype=float)
        self.fovy = np.array([float(self.model.cam_fovy[self.camera_id] or DEFAULT_FOVY)], dtype=float)
        self.running = True
        self.lock = threading.Lock()
        self._apply_pose_locked()

    def _make_tuning_scene_visible(self) -> None:
        """Make the preview useful without changing the real XML scene."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("coffee_machine_"):
                self.model.geom_rgba[geom_id] = np.array([0.35, 0.35, 0.35, 1.0])
            if "gripper_camera_body" in name or "gripper_camera_lens" in name:
                self.model.geom_rgba[geom_id, 3] = 0.0
    def _current_quat(self) -> np.ndarray:
        delta = quat_from_euler_xyz(*np.deg2rad(self.delta_rpy_deg))
        q = quat_mul(self.base_quat, delta)
        return q / np.linalg.norm(q)

    def _apply_pose_locked(self) -> None:
        self.model.cam_pos[self.camera_id] = self.cam_pos
        self.model.cam_quat[self.camera_id] = self._current_quat()
        self.model.cam_fovy[self.camera_id] = float(self.fovy[0])
        mujoco.mj_forward(self.model, self.data)

    def print_xml_values(self) -> None:
        with self.lock:
            cam_pos = self.cam_pos.copy()
            q = self._current_quat()
            fovy = float(self.fovy[0])
            delta = self.delta_rpy_deg.copy()
        print("\nPaste this gripper camera element back into so101_gripper.xml:")
        print(
            f'<camera name="gripper_cam" mode="fixed" '
            f'pos="{cam_pos[0]:.4f} {cam_pos[1]:.4f} {cam_pos[2]:.4f}" '
            f'quat="{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}" '
            f'fovy="{fovy:.1f}"/>'
        )
        print(f"delta rpy_deg = {delta[0]:.1f}, {delta[1]:.1f}, {delta[2]:.1f}")

    def launch_viewer_and_preview(self) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while self.running and viewer.is_running():
                with self.lock:
                    self._apply_pose_locked()
                    frame = self.env.sim.render(
                        camera_name=CAMERA_NAME,
                        width=PREVIEW_W,
                        height=PREVIEW_H,
                        depth=False,
                    )
                    frame = np.flipud(frame)
                    viewer.sync()
                cv2.imshow(CAMERA_NAME, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                time.sleep(0.01)
        self.running = False

    def launch_sliders(self) -> None:
        root = tk.Tk()
        root.title("SO101 gripper camera view tuner")

        rows: list[tuple[str, np.ndarray, int, float, float, float]] = [
            ("cam X", self.cam_pos, 0, -0.050, 0.080, 0.001),
            ("cam Y", self.cam_pos, 1, -0.080, 0.080, 0.001),
            ("cam Z", self.cam_pos, 2, -0.080, 0.080, 0.001),
            ("roll d", self.delta_rpy_deg, 0, -180.0, 180.0, 1.0),
            ("pitch d", self.delta_rpy_deg, 1, -180.0, 180.0, 1.0),
            ("yaw d", self.delta_rpy_deg, 2, -180.0, 180.0, 1.0),
            ("fovy", self.fovy, 0, 20.0, 140.0, 1.0),
        ]

        def add_slider(row: int, label: str, target: np.ndarray, idx: int, lo: float, hi: float, step: float) -> None:
            ttk.Label(root, text=label, width=10).grid(row=row, column=0, padx=6, pady=4)
            value = tk.DoubleVar(value=float(target[idx]))
            entry = ttk.Entry(root, textvariable=value, width=10)
            entry.grid(row=row, column=2, padx=6)

            def update(raw: str | None = None) -> None:
                try:
                    val = round(float(value.get()) / step) * step
                except tk.TclError:
                    return
                with self.lock:
                    target[idx] = val

            scale = ttk.Scale(root, from_=lo, to=hi, variable=value, orient="horizontal", command=update, length=320)
            scale.grid(row=row, column=1, padx=6, pady=4)
            entry.bind("<Return>", lambda _event: update())

        for i, row in enumerate(rows):
            add_slider(i, *row)

        ttk.Button(root, text="Print XML values", command=self.print_xml_values).grid(
            row=len(rows), column=0, columnspan=3, sticky="ew", padx=6, pady=8
        )

        def on_close() -> None:
            self.running = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
        self.running = False

    def close(self) -> None:
        try:
            cv2.destroyWindow(CAMERA_NAME)
        except Exception:
            pass
        self.env.close()


def main() -> None:
    tuner = CameraViewTuner()
    viewer_thread = threading.Thread(target=tuner.launch_viewer_and_preview, daemon=True)
    viewer_thread.start()
    try:
        tuner.launch_sliders()
    finally:
        tuner.running = False
        tuner.close()


if __name__ == "__main__":
    main()