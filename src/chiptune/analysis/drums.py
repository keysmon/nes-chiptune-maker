"""Drum onset detection and kick/snare/hat classification.

The NES noise channel is unpitched, so drum "pitch" is a placeholder; what
matters is which of the three drum voices (see config `[drums]`) a hit maps
to, decided here from the hit's spectral centroid: low-energy-in-bass ->
KICK, bright/broadband -> HAT, everything between -> SNARE.

Centroid is computed with `n_fft`/`hop_length` pinned to the analysis
window's own length and `center=False`. Calling
`librosa.feature.spectral_centroid` on a short window with its defaults
(`n_fft=2048`, `center=True`) zero-pads and reflection-pads a ~40ms window
out to a much longer effective frame, which on a real 60 Hz kick burst
produced a wildly unstable centroid (194 Hz-582 Hz depending on window
length, sometimes crossing the kick/snare boundary). Pinning both to the
window length makes the whole window exactly one Hann-tapered STFT frame,
which measured a stable ~75-103 Hz across window lengths 20-60ms for the
same burst - a clean 2 octaves below the snare boundary.
"""
from __future__ import annotations

import sys

import numpy as np
import librosa

from chiptune.score import NoteEvent, Percussion, Role

WINDOW_SECONDS = 0.04
NOTE_DURATION = 0.05
PLACEHOLDER_PITCH = 38  # noise channel has no pitch; MIDI 38 = acoustic snare, for readability only
MIN_WINDOW_SAMPLES = 8  # below this, an FFT centroid isn't meaningful


def _hit_centroid(window: np.ndarray, sr: int) -> float:
    centroid = librosa.feature.spectral_centroid(
        y=window, sr=sr, n_fft=len(window), hop_length=len(window), center=False
    )
    return float(centroid.mean())


def _classify(centroid: float, kick_max_hz: float, hat_min_hz: float) -> Percussion:
    if centroid < kick_max_hz:
        return Percussion.KICK
    if centroid > hat_min_hz:
        return Percussion.HAT
    return Percussion.SNARE


def transcribe_drums(
    stem: np.ndarray, sr: int, kick_max_hz: float, hat_min_hz: float, backtrack: bool = True
) -> list[NoteEvent]:
    """Detect onsets in a drum stem and classify each into KICK/SNARE/HAT.

    Emits `NoteEvent`s with `role=PERCUSSION`, a nominal `NOTE_DURATION`, and
    a placeholder pitch (the noise channel is unpitched; the `percussion`
    kind is what actually selects the drum voice downstream).
    """
    onsets = librosa.onset.onset_detect(y=stem, sr=sr, units="samples", backtrack=backtrack)
    window_len = max(int(WINDOW_SECONDS * sr), MIN_WINDOW_SAMPLES)

    notes = []
    counts = {Percussion.KICK: 0, Percussion.SNARE: 0, Percussion.HAT: 0}
    for onset in onsets:
        window = stem[onset : onset + window_len]
        if len(window) < MIN_WINDOW_SAMPLES:
            continue  # onset too close to the end of the stem for a meaningful window

        centroid = _hit_centroid(window, sr)
        kind = _classify(centroid, kick_max_hz, hat_min_hz)
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
