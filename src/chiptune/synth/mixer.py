"""NES non-linear output mixer and WAV writer.

The chip does not sum its channels linearly. The two pulse channels share one
non-linear DAC and the triangle/noise/DPCM group shares another. This curve is
why loud NES music compresses instead of clipping, and it is audible - a linear
sum sounds noticeably harsher.

Formulas are the NESdev-documented approximations, with dmc = 0 (DPCM is out of
scope per the spec).

The hardware curve is defined for non-negative DAC levels, but the APU feeds this
mixer zero-mean, band-limited channel waveforms (the pulse Fourier series drops
its DC term, the triangle table is symmetric about 0, noise is +/-1). We therefore
apply the curve odd-symmetrically: compress the magnitude and restore the sign.

This matters because the naive guard ``np.where(x > 0, curve, 0)`` half-wave
rectifies a zero-mean input - it clamps the entire negative half to 0. On a
band-limited waveform that both injects a positive DC offset and folds new high
harmonics back in (a buzzy octave-up artifact), worst on the triangle, which is
the bass channel. Odd-symmetric compression keeps a zero-mean input zero-mean and
treats both polarities identically, so it is exact for the non-negative case the
formula was derived for while behaving correctly on our bipolar signals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _compress(signal: np.ndarray, numerator: float, divisor_k: float) -> np.ndarray:
    """Apply the NES DAC compression curve odd-symmetrically.

    The hardware formula ``numerator / (divisor_k / level + 100)`` is defined for
    non-negative DAC levels. Our channels are bipolar zero-mean waveforms, so we
    apply the curve to ``|signal|`` and restore the sign. This preserves the
    curve's compression character for both polarities without half-wave-rectifying
    (which would add DC and a buzzy octave-up artifact, worst on the triangle bass
    channel). For non-negative input it is identical to ``np.where(x > 0, ..., 0)``.
    """
    mag = np.abs(signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        compressed = np.where(
            mag > 0.0,
            numerator / (divisor_k / np.where(mag > 0.0, mag, 1.0) + 100.0),
            0.0,
        )
    return np.sign(signal) * compressed


def nes_mix(
    pulse1: np.ndarray,
    pulse2: np.ndarray,
    triangle: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    lengths = {len(pulse1), len(pulse2), len(triangle), len(noise)}
    if len(lengths) != 1:
        raise ValueError(f"all channels must be the same length, got {sorted(lengths)}")

    pulse_out = _compress(pulse1 + pulse2, 95.88, 8128.0)
    tnd_out = _compress(triangle / 8227.0 + noise / 12241.0, 159.79, 1.0)  # dmc / 22638 == 0

    out = pulse_out + tnd_out
    return np.clip(out, -1.0, 1.0)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate, subtype="PCM_16")
