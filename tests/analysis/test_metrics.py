"""Fast unit test for the chroma-cosine "same song?" metric."""
import numpy as np
import librosa

from chiptune.analysis.metrics import chroma_cosine


def test_chroma_cosine_self_high_tritone_low():
    sr = 22050
    t = np.arange(int(1.5 * sr)) / sr
    # a note plus an octave harmonic -> a clean single-pitch-class chroma
    y = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=6)  # tritone: opposite pitch class

    self_sim = chroma_cosine(y, sr, y, sr)
    tritone_sim = chroma_cosine(y, sr, shifted, sr)
    assert self_sim > 0.99
    assert tritone_sim < self_sim - 0.3  # clearly less similar
    assert tritone_sim < 0.6
