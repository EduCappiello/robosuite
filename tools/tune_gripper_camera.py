"""Unified interactive tuner for the SO101 wrist camera assembly.

Run from the robosuite repository root:

    python tools/tune_gripper_camera.py --target mount
    python tools/tune_gripper_camera.py --target sensor
    python tools/tune_gripper_camera.py --target view
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np
import robosuite as suite
import robosuite_private  # noqa: F401 - registers private envs / robots


@dataclass(frozen=True)
class BodyTarget:
    body_name: str
    xml_name: str
    label: str
    pos: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


BODY_TARGETS = {
    "mount": BodyTarget(
        body_name="robot0_gripper_camera_mount",
        xml_name="gripper_camera_mount",
        label="camera mount",
        pos=(0.062, 0.058, -0.060),
        rpy_deg=(0.0, 88.0, 0.0),
    ),
    "sensor": BodyTarget(
        body_name="robot0_gripper_camera_sensor",
        xml_name="gripper_camera_sensor",
        label="camera sensor",
        pos=(-0.037, -0.056, -0.013),
        rpy_deg=(-1.0, 13.0, 0.0),
    ),
}


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


class CameraBodyTuner:
    def __init__(self, target_name: str) -> None:
        self.target = BODY_TARGETS[target_name]
        self.env = suite.make(
            env_name="SOARM101PnPCup",
            robots="SOARM101",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=[],
        )
        self.env.reset()
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target.body_name)
        if self.body_id < 0:
            raise RuntimeError(f"Could not find body {self.target.body_name!r}. Check so101_gripper.xml.")

        self.pos = np.array(self.target.pos, dtype=float)
        self.rpy_deg = np.array(self.target.rpy_deg, dtype=float)
        self.running = True
        self.lock = threading.Lock()
        self._apply_pose_locked()

    def _current_quat(self) -> np.ndarray:
        return quat_from_euler_xyz(*np.deg2rad(self.rpy_deg))

    def _apply_pose_locked(self) -> None:
        self.model.body_pos[self.body_id] = self.pos
        self.model.body_quat[self.body_id] = self._current_quat()
        mujoco.mj_forward(self.model, self.data)

    def print_xml_values(self) -> None:
        with self.lock:
            pos = self.pos.copy()
            rpy_deg = self.rpy_deg.copy()
        quat = quat_from_euler_xyz(*np.deg2rad(rpy_deg))
        print(f"\nPaste this {self.target.label} body back into so101_gripper.xml:")
        print(
            f'<body name="{self.target.xml_name}" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'quat="{quat[0]:.7f} {quat[1]:.7f} {quat[2]:.7f} {quat[3]:.7f}">'
        )
        print(f"rpy_deg = {rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}")

    def launch_viewer(self) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while self.running and viewer.is_running():
                with self.lock:
                    self._apply_pose_locked()
                    viewer.sync()
                time.sleep(0.01)
        self.running = False

    def launch_sliders(self) -> None:
        root = tk.Tk()
        root.title(f"SO101 {self.target.label} tuner")
        rows = [
            ("X", self.pos, 0, -0.150, 0.150, 0.001),
            ("Y", self.pos, 1, -0.150, 0.150, 0.001),
            ("Z", self.pos, 2, -0.150, 0.150, 0.001),
            ("roll deg", self.rpy_deg, 0, -180.0, 180.0, 1.0),
            ("pitch deg", self.rpy_deg, 1, -180.0, 180.0, 1.0),
            ("yaw deg", self.rpy_deg, 2, -180.0, 180.0, 1.0),
        ]

        def add_slider(row, label, target, idx, lo, hi, step):
            ttk.Label(root, text=label, width=10).grid(row=row, column=0, padx=6, pady=4)
            value = tk.DoubleVar(value=float(target[idx]))
            ttk.Entry(root, textvariable=value, width=10).grid(row=row, column=2, padx=6)

            def update(_raw=None):
                try:
                    val = round(float(value.get()) / step) * step
                except tk.TclError:
                    return
                with self.lock:
                    target[idx] = val

            ttk.Scale(
                root, from_=lo, to=hi, variable=value, orient="horizontal", command=update, length=340
            ).grid(row=row, column=1, padx=6, pady=4)

        for index, row in enumerate(rows):
            add_slider(index, *row)

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
        self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the SO101 gripper camera assembly in MuJoCo.")
    parser.add_argument(
        "--target",
        choices=("mount", "sensor", "view"),
        required=True,
        help="mount=white STL bracket, sensor=PCB/lens assembly, view=optical camera/FOV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target == "view":
        from tune_gripper_camera_view import CameraViewTuner

        tuner = CameraViewTuner()
        viewer_target = tuner.launch_viewer_and_preview
    else:
        tuner = CameraBodyTuner(args.target)
        viewer_target = tuner.launch_viewer

    viewer_thread = threading.Thread(target=viewer_target, daemon=True)
    viewer_thread.start()
    try:
        tuner.launch_sliders()
    finally:
        tuner.running = False
        tuner.close()


if __name__ == "__main__":
    main()
