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
from scipy.signal import butter, sosfiltfilt


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
    # over: a tiny-but-nonzero |signal| makes divisor_k/mag overflow to inf, then
    # numerator/inf -> 0.0, which the outer np.where discards anyway. All three are
    # the correct limits, so silence the cosmetic warnings.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
    """Mix the four channels through the NES non-linear DAC and return the RAW signal.

    Deliberately does NOT clamp. The spec-6.1 "no clipping" invariant is only
    meaningful on the un-clamped signal, so returning the raw mix lets
    ``invariants.check_invariants`` actually see - and fail on - a genuine clip
    instead of a value the mixer silently distorted to fit [-1, 1]. The clamp to
    [-1, 1] happens once, at WAV-write time in the CLI, guarding only
    float-epsilon overshoot on an already-validated signal.
    """
    lengths = {len(pulse1), len(pulse2), len(triangle), len(noise)}
    if len(lengths) != 1:
        raise ValueError(f"all channels must be the same length, got {sorted(lengths)}")

    pulse_out = _compress(pulse1 + pulse2, 95.88, 8128.0)
    tnd_out = _compress(triangle / 8227.0 + noise / 12241.0, 159.79, 1.0)  # dmc / 22638 == 0

    return pulse_out + tnd_out


def apply_output_filter(
    signal: np.ndarray,
    sample_rate: int,
    highpass_hz: float = 0.0,
    lowpass_hz: float = 0.0,
) -> np.ndarray:
    """Gentle output-stage filtering on the final mixed signal.

    A real console's output stage was not flat: it cut sub-sonic DC rumble and
    rolled off the extreme top end. Raw ``nes_mix`` output is brighter/hotter
    than that, so this filters the finished mix to match - a high-pass at
    `highpass_hz` and a low-pass at `lowpass_hz`, kept gentle (2nd-order,
    zero-phase via sosfiltfilt) so the melody is not dulled. Same technique as
    the noise-channel low-pass in apu.py, for the same reason: filtfilt keeps
    transient timing exact since it does not shift phase.

    Either stage is skipped when its cutoff is 0 or outside the valid
    (0, Nyquist) range, so a cutoff of 0 disables that stage.
    """
    out = signal
    nyquist = sample_rate / 2.0
    if out.any():
        if 0 < highpass_hz < nyquist:
            sos = butter(2, highpass_hz, btype="high", fs=sample_rate, output="sos")
            out = sosfiltfilt(sos, out)
        if 0 < lowpass_hz < nyquist:
            sos = butter(2, lowpass_hz, btype="low", fs=sample_rate, output="sos")
            out = sosfiltfilt(sos, out)
    return out


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf  # lazy: the serverless demo has no libsndfile and writes WAV via stdlib instead
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate, subtype="PCM_16")
