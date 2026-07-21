"""Fast unit tests for the segment-wise chroma-cosine "same song?" metric.

The old mean-pooled-over-the-whole-track version of this metric didn't
discriminate: cosine similarity of two non-negative 12-vectors is inherently
high, so it mostly measured key, not song - two different same-key songs
would score 0.9+. These tests build a signal with a time-varying pitch-class
progression (four distinct tones in sequence) specifically to prove the
reworked, time-resolved metric is sensitive to *when* pitch classes occur,
not just which ones are present overall.
"""
import numpy as np
import librosa

from chiptune.analysis.metrics import chroma_cosine, CHROMA_WINDOW_SECONDS


def _progression(sr: int) -> np.ndarray:
    """Four distinct-pitch-class tones, one per `CHROMA_WINDOW_SECONDS` window."""
    midi_notes = [60, 64, 67, 71]  # C, E, G, B - four different pitch classes
    segments = []
    for midi in midi_notes:
        freq = librosa.midi_to_hz(midi)
        t = np.arange(int(CHROMA_WINDOW_SECONDS * sr)) / sr
        segments.append((0.6 * np.sin(2 * np.pi * freq * t)).astype("float32"))
    return np.concatenate(segments)


def test_chroma_cosine_self_is_near_one():
    sr = 22050
    y = _progression(sr)
    assert chroma_cosine(y, sr, y, sr) > 0.99


def test_chroma_cosine_time_shuffled_scores_below_self():
    # Same four pitch classes present overall (so the OLD track-wide-mean
    # metric can't tell these apart) but reordered in time, which destroys
    # the progression a time-resolved comparison should penalize.
    sr = 22050
    y = _progression(sr)
    win = int(CHROMA_WINDOW_SECONDS * sr)
    segments = [y[i * win : (i + 1) * win] for i in range(4)]
    shuffled = np.concatenate([segments[1], segments[3], segments[0], segments[2]])  # derangement

    self_sim = chroma_cosine(y, sr, y, sr)
    shuffled_sim = chroma_cosine(y, sr, shuffled, sr)
    assert shuffled_sim < self_sim - 0.3  # clearly less similar
    assert shuffled_sim < 0.6


def test_chroma_cosine_tritone_transposed_scores_below_self():
    sr = 22050
    y = _progression(sr)
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=6)  # tritone: opposite pitch class throughout

    self_sim = chroma_cosine(y, sr, y, sr)
    tritone_sim = chroma_cosine(y, sr, shifted, sr)
    assert tritone_sim < self_sim - 0.3
    assert tritone_sim < 0.6


def test_chroma_cosine_silence_is_zero():
    sr = 22050
    y = _progression(sr)
    silence = np.zeros_like(y)
    assert chroma_cosine(y, sr, silence, sr) == 0.0
