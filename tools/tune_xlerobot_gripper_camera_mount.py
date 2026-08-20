"""Interactively tune the XLeRobot right-gripper camera assembly.

Run from the robosuite repository root:

    python tools/tune_xlerobot_gripper_camera_mount.py
    python tools/tune_xlerobot_gripper_camera_mount.py --target camera

The sliders update either the white bracket or its child camera body. Press
``Print XML values`` and paste the resulting pose back into XLeRobot robot.xml.
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
import robosuite_private  # noqa: F401 - registers private environments


# Viewer-only wrist pose. These values never get written to the robot XML.
PREVIEW_WRIST_DEG = (0.0, 0.0)
PREVIEW_TASK_RIGHT_POSES = {
    "task1": (-23.03, -104.09, 79.38, 27.56, 82.33, 1.33),
    "task3": (-36.75, -104.79, 24.53, 96.40, 86.20, 1.06),
    "task5": (-50.99, -104.44, 52.13, 94.64, 85.76, 1.06),
}


@dataclass(frozen=True)
class Target:
    body_name: str
    xml_body_name: str
    label: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]


TARGETS = {
    "mount": Target(
        body_name="robot0_right_gripper_camera_mount",
        xml_body_name="right_gripper_camera_mount",
        label="camera mount",
        pos=(0.001, 0.0255, -0.0845),
        quat=(0.47788572, 0.52150511, -0.51697386, 0.48207436),
    ),
    "camera": Target(
        body_name="robot0_right_gripper_camera_sensor",
        xml_body_name="right_gripper_camera_sensor",
        label="camera sensor",
        pos=(0.0545, 0.001, 0.090),
        quat=(0.0, -0.31316381, 0.0, 0.94969913),
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


def euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.rad2deg((roll, pitch, yaw))


class MountTuner:
    def __init__(self, target_name: str, task_name: str) -> None:
        self.target = TARGETS[target_name]
        self.env = suite.make(
            env_name=f"cupPnP_{task_name}",
            robots=["XLeRobot"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names=[],
        )
        self.env.reset()
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data
        task_pose = PREVIEW_TASK_RIGHT_POSES.get(task_name)
        self.preview_wrist_deg = np.array(
            task_pose[3:5] if task_pose is not None else PREVIEW_WRIST_DEG,
            dtype=float,
        )
        if task_pose is not None:
            for joint_suffix, angle_deg in zip(
                ("shoulder_pan", "shoulder_lift", "elbow_flex"),
                task_pose[:3],
                strict=True,
            ):
                joint = self.model.joint(f"robot0_right_{joint_suffix}")
                low, high = (float(value) for value in joint.range)
                self.data.qpos[joint.qposadr[0]] = np.clip(
                    math.radians(angle_deg), low + 0.005, high - 0.005
                )

            gripper_joint = self.model.joint("robot0_right_gripper")
            low, high = (float(value) for value in gripper_joint.range)
            self.data.qpos[gripper_joint.qposadr[0]] = low + task_pose[5] / 100.0 * (high - low)
        self.wrist_qpos_addresses = []
        self.wrist_ranges = []
        for joint_name in ("robot0_right_wrist_flex", "robot0_right_wrist_roll"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Could not find XLeRobot joint {joint_name!r}.")
            self.wrist_qpos_addresses.append(self.model.jnt_qposadr[joint_id])
            self.wrist_ranges.append(tuple(float(value) for value in self.model.jnt_range[joint_id]))

        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.target.body_name
        )
        if self.body_id < 0:
            raise RuntimeError(f"Could not find body {self.target.body_name!r} in the compiled model.")

        self.pos = np.array(self.target.pos, dtype=float)
        self.rpy_deg = euler_xyz_from_quat(np.array(self.target.quat, dtype=float))
        self.running = True
        self.lock = threading.Lock()
        self._apply_pose_locked()

    def _quat(self) -> np.ndarray:
        return quat_from_euler_xyz(*np.deg2rad(self.rpy_deg))

    def _apply_pose_locked(self) -> None:
        for qpos_address, angle_deg, (low, high) in zip(
            self.wrist_qpos_addresses,
            self.preview_wrist_deg,
            self.wrist_ranges,
            strict=True,
        ):
            self.data.qpos[qpos_address] = np.clip(
                math.radians(angle_deg), low + 0.005, high - 0.005
            )
        self.model.body_pos[self.body_id] = self.pos
        self.model.body_quat[self.body_id] = self._quat()
        mujoco.mj_forward(self.model, self.data)

    def print_xml_values(self) -> None:
        with self.lock:
            pos = self.pos.copy()
            rpy_deg = self.rpy_deg.copy()
            quat = self._quat()
        print(f"\nPaste this {self.target.label} body back into XLeRobot/robot.xml:")
        print(
            f'<body name="{self.target.xml_body_name}" '
            f'pos="{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}" '
            f'quat="{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}">'
        )
        print(
            f"{self.target.label} rpy_deg = "
            f"{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}"
        )

    def launch_viewer(self) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Collision geoms use group 0; visual meshes and the white bracket use
            # group 1. Hide collisions only in this tuning viewer.
            viewer.opt.geomgroup[0] = 0
            viewer.opt.geomgroup[1] = 1
            while self.running and viewer.is_running():
                with self.lock:
                    self._apply_pose_locked()
                    viewer.sync()
                time.sleep(0.01)
        self.running = False

    def launch_sliders(self) -> None:
        root = tk.Tk()
        root.title(f"XLeRobot right gripper {self.target.label} tuner")
        rows = [
            ("X", self.pos, 0, -0.20, 0.20, 0.0005),
            ("Y", self.pos, 1, -0.20, 0.20, 0.0005),
            ("Z", self.pos, 2, -0.20, 0.20, 0.0005),
            ("roll deg", self.rpy_deg, 0, -180.0, 180.0, 0.5),
            ("pitch deg", self.rpy_deg, 1, -180.0, 180.0, 0.5),
            ("yaw deg", self.rpy_deg, 2, -180.0, 180.0, 0.5),
            ("wrist flex", self.preview_wrist_deg, 0, -95.0, 95.0, 1.0),
            ("wrist roll", self.preview_wrist_deg, 1, -157.0, 162.0, 1.0),
        ]

        for row, (label, target, index, low, high, step) in enumerate(rows):
            ttk.Label(root, text=label, width=10).grid(row=row, column=0, padx=6, pady=4)
            value = tk.DoubleVar(value=float(target[index]))
            ttk.Entry(root, textvariable=value, width=12).grid(row=row, column=2, padx=6)

            def update(_raw=None, *, value=value, target=target, index=index, step=step):
                try:
                    new_value = round(float(value.get()) / step) * step
                except tk.TclError:
                    return
                with self.lock:
                    target[index] = new_value

            ttk.Scale(
                root,
                from_=low,
                to=high,
                variable=value,
                orient="horizontal",
                command=update,
                length=360,
            ).grid(row=row, column=1, padx=6, pady=4)

        ttk.Button(root, text="Print XML values", command=self.print_xml_values).grid(
            row=len(rows), column=0, columnspan=3, sticky="ew", padx=6, pady=8
        )

        def close() -> None:
            self.running = False
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close)
        root.mainloop()
        self.running = False

    def close(self) -> None:
        self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, default="mount")
    parser.add_argument(
        "--task",
        choices=("task1", "task2", "task3", "task5"),
        default="task1",
        help="Scene to preview; Task 1/3/5 also load their fixed right-arm pose.",
    )
    args = parser.parse_args()
    tuner = MountTuner(args.target, args.task)
    viewer_thread = threading.Thread(target=tuner.launch_viewer, daemon=True)
    viewer_thread.start()
    try:
        tuner.launch_sliders()
    finally:
        tuner.running = False
        tuner.close()


if __name__ == "__main__":
    main()
