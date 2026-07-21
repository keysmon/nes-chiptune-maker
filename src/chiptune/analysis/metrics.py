"""Chroma-cosine similarity: a timbre-robust proxy for "is this the same song?".

Mean-pooled 12-bin chroma (pitch-class energy) cosine similarity. Chroma folds
all octaves into 12 pitch classes and ignores spectral shape, so it looks past
the enormous timbral gap between a real recording and a 4-voice NES chip render
and compares only harmonic/pitch-class content. A faithful chiptune scores high;
a wrong-key or wrong-notes render scores low. This is the one automated compass
for tuning the analysis pipeline, so `convert` prints it alongside the artifact.
"""
from __future__ import annotations

import numpy as np
import librosa


def _mean_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return chroma.mean(axis=1)


def chroma_cosine(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
    """Cosine similarity of the mean-pooled chroma vectors of `a` and `b`.

    Returns 0.0 if either signal has no chroma energy (silence).
    """
    ca = _mean_chroma(a, sr_a)
    cb = _mean_chroma(b, sr_b)
    denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(ca, cb) / denom)
