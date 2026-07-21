"""Drum onset detection and kick/snare/hat classification.

The NES noise channel is unpitched, so drum "pitch" is a placeholder; what
matters is which of the three drum voices (see config `[drums]`) a hit maps to.

Classification is by LOW-BAND / HIGH-BAND ENERGY FRACTION, not spectral
centroid. On a *summed* drum stem (kick + snare + hats mixed into one signal,
as Demucs produces) a kick's onset window also carries hat/cymbal energy
bleeding from nearby overlapping hits; that bright bleed drags the centroid up
~4x and collapses the split - on the real pop stem the centroid classifier
measured kick=0, snare=112, hat=11. The energy *fraction* below `kick_band_hz`
is robust to that bleed: a kick still puts most of its energy in the sub-band
even with a hat ringing on top, so the fraction recovers a real kick/snare/hat
split (measured ~46/48/29 on the same stem).

Each onset window is Hann-tapered before the rfft so low-band energy does not
leak into the high band and inflate the hat fraction.
"""
from __future__ import annotations

import sys

import numpy as np
import librosa

from chiptune.score import NoteEvent, Percussion, Role

WINDOW_SECONDS = 0.04
NOTE_DURATION = 0.05
PLACEHOLDER_PITCH = 38  # noise channel has no pitch; MIDI 38 = acoustic snare, for readability only
MIN_WINDOW_SAMPLES = 8  # below this, an FFT band split isn't meaningful


def _band_fractions(
    window: np.ndarray, sr: int, kick_band_hz: float, hat_band_hz: float
) -> tuple[float, float]:
    """Return (low_frac, high_frac): the fraction of window power below
    `kick_band_hz` and above `hat_band_hz`. Both 0.0 for a silent window."""
    tapered = window * np.hanning(len(window))
    power = np.abs(np.fft.rfft(tapered)) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0, 0.0
    freqs = np.fft.rfftfreq(len(window), 1.0 / sr)
    low = float(power[freqs < kick_band_hz].sum()) / total
    high = float(power[freqs > hat_band_hz].sum()) / total
    return low, high


def _classify(
    low_frac: float, high_frac: float, kick_low_frac_min: float, hat_high_frac_min: float
) -> Percussion:
    if low_frac >= kick_low_frac_min:
        return Percussion.KICK
    if high_frac >= hat_high_frac_min:
        return Percussion.HAT
    return Percussion.SNARE


def transcribe_drums(
    stem: np.ndarray,
    sr: int,
    kick_band_hz: float,
    hat_band_hz: float,
    kick_low_frac_min: float,
    hat_high_frac_min: float,
    backtrack: bool = True,
) -> list[NoteEvent]:
    """Detect onsets in a drum stem and classify each into KICK/SNARE/HAT.

    Emits `NoteEvent`s with `role=PERCUSSION`, a nominal `NOTE_DURATION`, and a
    placeholder pitch (the noise channel is unpitched; the `percussion` kind is
    what actually selects the drum voice downstream).
    """
    onsets = librosa.onset.onset_detect(y=stem, sr=sr, units="samples", backtrack=backtrack)
    window_len = max(int(WINDOW_SECONDS * sr), MIN_WINDOW_SAMPLES)

    notes = []
    counts = {Percussion.KICK: 0, Percussion.SNARE: 0, Percussion.HAT: 0}
    for onset in onsets:
        window = stem[onset : onset + window_len]
        if len(window) < MIN_WINDOW_SAMPLES:
            continue  # onset too close to the end of the stem for a meaningful window

        low_frac, high_frac = _band_fractions(window, sr, kick_band_hz, hat_band_hz)
        kind = _classify(low_frac, high_frac, kick_low_frac_min, hat_high_frac_min)
        counts[kind] += 1

        velocity = int(np.clip(np.abs(window).max() * 127, 1, 127))
        start = float(onset) / sr
        notes.append(
            NoteEvent(
                pitch=PLACEHOLDER_PITCH,
                start=start,
                end=start + NOTE_DURATION,
                velocity=velocity,
                role=Role.PERCUSSION,
                percussion=kind,
            )
        )

    print(
        f"chiptune.analysis.drums: classified {len(notes)} hits - "
        f"kick={counts[Percussion.KICK]} snare={counts[Percussion.SNARE]} hat={counts[Percussion.HAT]}",
        file=sys.stderr,
    )
    return notes
