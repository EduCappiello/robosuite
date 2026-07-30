"""Interactive tuner for SOARM101 initial joint pose.

Run from the robosuite repo root:
    conda run -n lerobot python tools/tune_soarm101_init_qpos.py

The sliders set the robot arm qpos directly in MuJoCo, so you can choose a reset
pose where the wrist camera sees the coffee machine / cup area. Press
"Print init_qpos" and paste the printed array into soarm101_robot.py.
"""

from __future__ import annotations

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


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
DEFAULT_QPOS_RAD = np.array([0.0875, -1.8174, 1.5443, 1.3579, 1.6809], dtype=float)
CAMERA_NAME = "robot0_gripper_cam"
PREVIEW_W = 320
PREVIEW_H = 240


class InitPoseTuner:
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
        self.robot = self.env.robots[0]
        self.qpos_indexes = list(self.robot._ref_joint_pos_indexes[: len(MOTOR_NAMES)])

        self.qpos_deg = np.rad2deg(DEFAULT_QPOS_RAD).astype(float)
        self.running = True
        self.lock = threading.Lock()
        self._apply_pose_locked()

    def _apply_pose_locked(self) -> None:
        qpos_rad = np.deg2rad(self.qpos_deg)
        for idx, value in zip(self.qpos_indexes, qpos_rad):
            self.data.qpos[idx] = value
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def print_init_qpos(self) -> None:
        with self.lock:
            qpos_deg = self.qpos_deg.copy()
            qpos_rad = np.deg2rad(qpos_deg)
        print("\nPaste this init_qpos back into soarm101_robot.py:")
        print(
            "return np.array(["
            + ", ".join(f"{v:.4f}" for v in qpos_rad)
            + "])"
        )
        print(
            "degrees = "
            + ", ".join(f"{name}={deg:.2f}" for name, deg in zip(MOTOR_NAMES, qpos_deg))
        )

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
        root.title("SOARM101 init qpos tuner")

        ranges = {
            "shoulder_pan": (-120.0, 120.0),
            "shoulder_lift": (-130.0, 40.0),
            "elbow_flex": (-20.0, 140.0),
            "wrist_flex": (-100.0, 120.0),
            "wrist_roll": (-180.0, 180.0),
        }

        def add_slider(row: int, name: str) -> None:
            ttk.Label(root, text=name, width=14).grid(row=row, column=0, padx=6, pady=4)
            idx = MOTOR_NAMES.index(name)
            value = tk.DoubleVar(value=float(self.qpos_deg[idx]))
            entry = ttk.Entry(root, textvariable=value, width=10)
            entry.grid(row=row, column=2, padx=6)

            def update(raw: str | None = None) -> None:
                try:
                    val = round(float(value.get()) * 10.0) / 10.0
                except tk.TclError:
                    return
                with self.lock:
                    self.qpos_deg[idx] = val

            lo, hi = ranges[name]
            scale = ttk.Scale(root, from_=lo, to=hi, variable=value, orient="horizontal", command=update, length=360)
            scale.grid(row=row, column=1, padx=6, pady=4)
            entry.bind("<Return>", lambda _event: update())

        for i, name in enumerate(MOTOR_NAMES):
            add_slider(i, name)

        ttk.Button(root, text="Print init_qpos", command=self.print_init_qpos).grid(
            row=len(MOTOR_NAMES), column=0, columnspan=3, sticky="ew", padx=6, pady=8
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
    tuner = InitPoseTuner()
    viewer_thread = threading.Thread(target=tuner.launch_viewer_and_preview, daemon=True)
    viewer_thread.start()
    try:
        tuner.launch_sliders()
    finally:
        tuner.running = False
        tuner.close()


if __name__ == "__main__":
    main()