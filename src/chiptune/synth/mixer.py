"""NES non-linear output mixer and WAV writer.

The chip does not sum its channels linearly. The two pulse channels share one
non-linear DAC and the triangle/noise/DPCM group shares another. This curve is
why loud NES music compresses instead of clipping, and it is audible - a linear
sum sounds noticeably harsher.

Formulas are the NESdev-documented approximations, with dmc = 0 (DPCM is out of
scope per the spec).

The APU feeds this mixer zero-mean, band-limited channel waveforms scaled into
the chip's 0-15 DAC range. A real pulse channel only ever presents the DAC a
non-negative level (the volume during the duty-high portion, 0 during duty-low),
so the ``p_sum > 0`` guard is not merely a divide-by-zero shield: for a zero-mean
pulse it clamps the duty-low half to 0, reconstructing exactly the unipolar DAC
input the formula is derived for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def nes_mix(
    pulse1: np.ndarray,
    pulse2: np.ndarray,
    triangle: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    lengths = {len(pulse1), len(pulse2), len(triangle), len(noise)}
    if len(lengths) != 1:
        raise ValueError(f"all channels must be the same length, got {sorted(lengths)}")

    with np.errstate(divide="ignore", invalid="ignore"):
        p_sum = pulse1 + pulse2
        pulse_out = np.where(
            p_sum > 0.0,
            95.88 / (8128.0 / np.where(p_sum > 0.0, p_sum, 1.0) + 100.0),
            0.0,
        )

        tnd = triangle / 8227.0 + noise / 12241.0     # dmc / 22638 == 0
        tnd_out = np.where(
            tnd > 0.0,
            159.79 / (1.0 / np.where(tnd > 0.0, tnd, 1.0) + 100.0),
            0.0,
        )

    out = pulse_out + tnd_out
    return np.clip(out, -1.0, 1.0)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate, subtype="PCM_16")
