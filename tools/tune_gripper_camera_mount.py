"""Compatibility entry point for the unified SO101 camera tuner.

Prefer:
    python tools/tune_gripper_camera.py --target mount|sensor|view

This legacy filename still defaults to sensor when no --target is provided,
matching its previous behavior.
"""

from __future__ import annotations

import sys

from tune_gripper_camera import main


if __name__ == "__main__":
    if not any(arg == "--target" or arg.startswith("--target=") for arg in sys.argv[1:]):
        sys.argv.extend(["--target", "sensor"])
    main()