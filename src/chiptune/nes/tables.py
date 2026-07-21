"""Ricoh 2A03 (NTSC) frequency and period arithmetic.

The chip does not take frequencies. It takes 11-bit period registers, and the
resulting pitch is quantized by that integer division. Pitch error grows at high
frequencies where consecutive period values are far apart in Hz - this is a real
property of the hardware, not an approximation on our side.
"""
from __future__ import annotations

CPU_NTSC = 1789773           # Hz
PERIOD_MAX = 2047            # 11-bit register
PERIOD_MIN = 1

# NTSC noise period table, indexed 0-15 by the low nibble of $400E.
NOISE_PERIODS: tuple[int, ...] = (
    4, 8, 16, 32, 64, 96, 128, 160,
    202, 254, 380, 508, 762, 1016, 2034, 4068,
)


def midi_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _clamp_period(p: int) -> int:
    return max(PERIOD_MIN, min(PERIOD_MAX, p))


def pulse_period(freq_hz: float) -> int:
    """f = CPU / (16 * (period + 1))"""
    if freq_hz <= 0:
        raise ValueError(f"frequency must be positive (got {freq_hz})")
    return _clamp_period(int(round(CPU_NTSC / (16.0 * freq_hz) - 1.0)))


def pulse_period_to_hz(period: int) -> float:
    return CPU_NTSC / (16.0 * (period + 1))


def triangle_period(freq_hz: float) -> int:
    """f = CPU / (32 * (period + 1)). The /32 puts triangle an octave below pulse."""
    if freq_hz <= 0:
        raise ValueError(f"frequency must be positive (got {freq_hz})")
    return _clamp_period(int(round(CPU_NTSC / (32.0 * freq_hz) - 1.0)))


def triangle_period_to_hz(period: int) -> float:
    return CPU_NTSC / (32.0 * (period + 1))


def playable_on_pulse(pitch: int) -> bool:
    f = midi_to_hz(pitch)
    exact = CPU_NTSC / (16.0 * f) - 1.0
    return PERIOD_MIN <= exact <= PERIOD_MAX


def playable_on_triangle(pitch: int) -> bool:
    f = midi_to_hz(pitch)
    exact = CPU_NTSC / (32.0 * f) - 1.0
    return PERIOD_MIN <= exact <= PERIOD_MAX
