"""2A03 triangle channel: a 32-step, 4-bit staircase.

Deliberately NOT band-limited. The real chip outputs this staircase, and its
harmonics are part of the authentic sound. Band-limiting would make it sound
like a generic synth triangle instead of an NES triangle.

The hardware has no volume control on this channel - it is on or off.
"""
from __future__ import annotations

import numpy as np

# 0..15 then 15..0, mapped to [-1, 1]. Each of the 16 DAC levels is visited twice.
_STEPS = np.concatenate([np.arange(16), np.arange(15, -1, -1)]).astype(np.float64)
TRIANGLE_TABLE: np.ndarray = (_STEPS - 7.5) / 7.5

TABLE_SIZE = TRIANGLE_TABLE.shape[0]


def render_triangle(
    freq_hz: float,
    n_samples: int,
    sample_rate: int,
    phase: float,
) -> tuple[np.ndarray, float]:
    """Return (samples, next_phase). `phase` is in turns, i.e. [0, 1)."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64), phase
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive (got {freq_hz})")

    step = freq_hz / sample_rate
    phases = phase + step * np.arange(n_samples, dtype=np.float64)
    # Nearest-step lookup, no interpolation: the staircase is the point.
    idx = ((phases % 1.0) * TABLE_SIZE).astype(np.int64) % TABLE_SIZE
    return TRIANGLE_TABLE[idx], float((phase + step * n_samples) % 1.0)
