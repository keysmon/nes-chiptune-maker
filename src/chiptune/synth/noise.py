"""2A03 noise channel: a 15-bit linear feedback shift register.

Mode 0 ("long") taps bits 0 and 1, producing a 32767-step pseudo-random sequence
that reads as white noise. Mode 1 ("short") taps bits 0 and 6, producing a
93-step sequence short enough that the ear hears it as a tonal buzz.

Because each sequence is fixed and periodic, it is generated once and then
sampled at the rate implied by the period register. That is exact, not an
approximation, and it avoids stepping the register per output sample.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..nes.tables import CPU_NTSC, NOISE_PERIODS

_MODES = {"long": 1, "short": 6}


@lru_cache(maxsize=4)
def lfsr_sequence(mode: str) -> np.ndarray:
    """Full period of the shift register output, as +/-1 floats."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    tap = _MODES[mode]

    reg = 1
    seen: list[float] = []
    start = reg
    while True:
        feedback = (reg & 1) ^ ((reg >> tap) & 1)
        reg = (reg >> 1) | (feedback << 14)
        # Output is the inverse of bit 0.
        seen.append(1.0 if (reg & 1) == 0 else -1.0)
        if reg == start:
            break
    return np.asarray(seen, dtype=np.float64)


def render_noise(
    period_index: int,
    mode: str,
    n_samples: int,
    sample_rate: int,
    cursor: int,
) -> tuple[np.ndarray, int]:
    """Return (samples, next_cursor). `cursor` indexes into the LFSR sequence."""
    if not 0 <= period_index <= 15:
        raise ValueError(f"period_index must be 0-15, got {period_index}")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64), cursor

    seq = lfsr_sequence(mode)
    clock_hz = CPU_NTSC / NOISE_PERIODS[period_index]
    steps_per_sample = clock_hz / sample_rate

    positions = cursor + steps_per_sample * np.arange(n_samples, dtype=np.float64)
    idx = positions.astype(np.int64) % seq.shape[0]
    next_cursor = int(cursor + steps_per_sample * n_samples) % seq.shape[0]
    return seq[idx], next_cursor
