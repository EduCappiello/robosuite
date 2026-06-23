"""
apply_noise(signals, profile, rng) -> dict

Pure function — no global state, no side effects.
Returns a new dict; the input dict is not mutated.
"""

from __future__ import annotations

import numpy as np

from robosuite_private.noise.models import (
    BiasNoise,
    DropoutNoise,
    GaussianNoise,
    NoiseProfile,
    QuantizationNoise,
)


def apply_noise(
    signals: dict,
    profile: NoiseProfile,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Apply the noise models in `profile` to the matching keys in `signals`.

    Args:
        signals: dict of signal arrays (e.g. from read_motor_signals).
        profile: NoiseProfile describing which keys get which noise.
        rng:     numpy random Generator.  Defaults to a new Generator with
                 a random seed — pass a seeded generator for reproducibility.

    Returns:
        New dict with the same keys; noisy copies for matched keys,
        original references for unmatched keys (no copy).
    """
    if rng is None:
        rng = np.random.default_rng()

    out = dict(signals)  # shallow copy; we only replace matched keys

    for key, noise in profile.signals.items():
        if key not in signals:
            continue
        v = np.array(signals[key], dtype=np.float64)  # always copy

        if isinstance(noise, GaussianNoise):
            v = v + rng.normal(0.0, noise.std, size=v.shape).astype(np.float64)

        elif isinstance(noise, QuantizationNoise):
            r = float(noise.resolution)
            v = np.round(v / r) * r

        elif isinstance(noise, DropoutNoise):
            mask = rng.random(v.shape) < noise.rate
            v = np.where(mask, float(noise.fill_value), v)

        elif isinstance(noise, BiasNoise):
            v = v + np.asarray(noise.bias, dtype=np.float64)

        out[key] = v

    return out
