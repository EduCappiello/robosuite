"""
Noise profile dataclasses.

Each profile is a pure value object — it carries no state and no rng.
The rng is passed into apply_noise() so callers control seeding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianNoise:
    """Additive zero-mean Gaussian noise."""
    std: float | np.ndarray   # scalar or per-element std dev


@dataclass
class QuantizationNoise:
    """Round values to the nearest multiple of `resolution` (uniform ±½ bin)."""
    resolution: float         # [same units as the signal, e.g. rad or N·m]


@dataclass
class DropoutNoise:
    """Replace signal with `fill_value` with probability `rate`."""
    rate:       float         # dropout probability in [0, 1]
    fill_value: float = 0.0


@dataclass
class BiasNoise:
    """Constant additive offset (e.g. calibration error)."""
    bias: float | np.ndarray  # scalar or per-element bias


@dataclass
class NoiseProfile:
    """
    Per-key noise configuration passed to apply_noise().

    Keys match the signal dict returned by read_motor_signals() or
    get_ground_truth_dynamics().  Omit a key to pass that signal through
    unchanged.

    Example::

        profile = NoiseProfile(signals={
            "pos":    GaussianNoise(std=0.001),   # 1 mrad
            "vel_hw": GaussianNoise(std=0.01),    # 10 mrad/s
            "current": QuantizationNoise(resolution=0.05),
        })
    """
    signals: dict[str, GaussianNoise | QuantizationNoise | DropoutNoise | BiasNoise] = field(
        default_factory=dict
    )
