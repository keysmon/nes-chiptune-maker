"""Chroma-cosine similarity: a timbre-robust proxy for "is this the same song?".

Time-resolved (segment-wise) 12-bin chroma (pitch-class energy) cosine
similarity. Chroma folds all octaves into 12 pitch classes and ignores
spectral shape, so it looks past the enormous timbral gap between a real
recording and a 4-voice NES chip render and compares only harmonic/pitch-class
content.

A single track-wide mean-pooled chroma vector does NOT discriminate: cosine
similarity of two non-negative 12-vectors is inherently high (commonly ~0.75+
even for unrelated material, tending to 1.0 as either vector flattens), so a
track-wide pool mostly measures KEY, not SONG - two different same-key songs
would score 0.9+. Segmenting into fixed windows and averaging the per-window
cosine instead tracks the chord/pitch-class progression over time, which two
different same-key songs do not share, while a faithful same-song render does.

The window is `CHROMA_WINDOW_SECONDS` long: coarse enough to absorb the
chiptune's tempo-quantization jitter (a faithful render's note onsets land a
few tens of ms off the original, which would falsely penalize a strict
per-frame comparison), fine enough to capture a chord-progression change
within the track. This is the one automated compass for tuning the analysis
pipeline, so `convert` prints it alongside the artifact.
"""
from __future__ import annotations

import math

import numpy as np
import librosa

CHROMA_WINDOW_SECONDS = 2.0
_CHROMA_HOP_LENGTH = 512  # librosa.feature.chroma_cqt's default; kept explicit for the window-size math


def _windowed_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Chroma, mean-pooled into `CHROMA_WINDOW_SECONDS`-long windows.

    Returns an array of shape (n_windows, 12). A signal shorter than one
    window collapses to a single window covering the whole signal.
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_CHROMA_HOP_LENGTH)  # (12, n_frames)
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return np.zeros((0, 12), dtype=np.float32)

    frames_per_window = max(1, round(CHROMA_WINDOW_SECONDS * sr / _CHROMA_HOP_LENGTH))
    n_windows = max(1, math.ceil(n_frames / frames_per_window))
    # array_split (not a fixed-stride reshape) so a frame count that isn't an
    # exact multiple of the window size doesn't crash - it spreads the
    # remainder evenly across windows instead of leaving a ragged tail.
    return np.stack([split.mean(axis=1) for split in np.array_split(chroma, n_windows, axis=1)])


def chroma_cosine(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
    """Mean per-window chroma cosine similarity of `a` and `b`.

    Each signal is segmented into its own `CHROMA_WINDOW_SECONDS` windows
    (using its own sample rate - `a` and `b` need not share a sample rate),
    then aligned by window index over the shorter of the two window counts.
    A window pair is skipped if either side has no chroma energy (silence).
    Returns 0.0 if no window pair has energy on both sides.
    """
    wa = _windowed_chroma(a, sr_a)
    wb = _windowed_chroma(b, sr_b)
    n = min(len(wa), len(wb))

    sims = []
    for i in range(n):
        norm_a = float(np.linalg.norm(wa[i]))
        norm_b = float(np.linalg.norm(wb[i]))
        if norm_a == 0.0 or norm_b == 0.0:
            continue
        sims.append(float(np.dot(wa[i], wb[i]) / (norm_a * norm_b)))

    if not sims:
        return 0.0
    return float(np.mean(sims))
