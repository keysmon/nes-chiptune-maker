import numpy as np

from chiptune.score import TempoGrid
from chiptune.analysis.chords import detect_chords


def _sine_sum(freqs, t):
    y = np.zeros_like(t)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    return y / len(freqs)


def _synthetic_progression(sr, seconds_per_chord=1.0):
    """C major (C4 E4 G4) for one span, then A minor (A3 C4 E4) for another."""
    t = np.linspace(0, seconds_per_chord, int(sr * seconds_per_chord), endpoint=False)
    c_major = _sine_sum([261.63, 329.63, 392.00], t)   # C4 E4 G4 -> pitch classes 0, 4, 7
    a_minor = _sine_sum([220.00, 261.63, 329.63], t)   # A3 C4 E4 -> pitch classes 9, 0, 4
    return np.concatenate([c_major, a_minor]).astype(np.float32)


def test_detects_c_major_then_a_minor():
    sr = 22050
    mono = _synthetic_progression(sr, seconds_per_chord=1.0)
    # 120 BPM -> 0.5s/beat, so each 1s chord spans exactly 2 beats.
    grid = TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)

    segments = detect_chords(mono, sr, grid, smooth_beats=2)

    assert len(segments) == 2
    first, second = segments
    assert first.root == 0 and first.is_minor is False
    assert second.root == 9 and second.is_minor is True
    # Boundary should land near the true 1.0s split, tolerant to one beat (0.5s).
    assert abs(first.end - 1.0) <= 0.5
    assert abs(second.start - 1.0) <= 0.5
    assert first.start == 0.0
    assert second.end >= 1.5


def test_empty_signal_returns_no_segments():
    grid = TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)
    assert detect_chords(np.array([], dtype=np.float32), 22050, grid) == []


def test_micro_clip_returns_no_chords():
    """A signal shorter than one beat yields no spurious chord."""
    grid = TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)  # 0.5 s/beat
    y = np.zeros(int(22050 * 0.1), dtype=np.float32)  # 0.1 s << one beat
    assert detect_chords(y, 22050, grid) == []
