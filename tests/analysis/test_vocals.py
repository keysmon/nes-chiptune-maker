from pathlib import Path

import numpy as np
import pytest

from chiptune.analysis.vocals import transcribe_vocals
from chiptune.score import Role

CLIP = Path("out/spike_demucs/htdemucs/pop_src/vocals.wav")


def test_steady_tone_becomes_one_lead_note():
    sr = 22050
    f = 220.0  # A3 = MIDI 57
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * f * t)).astype("float32")
    notes = transcribe_vocals(y, sr, fmin=80, fmax=1000)
    assert notes and all(n.role is Role.LEAD for n in notes)
    # dominant note is A3 +/- 1 semitone
    longest = max(notes, key=lambda n: n.end - n.start)
    assert abs(longest.pitch - 57) <= 1


def test_silence_returns_no_notes():
    sr = 22050
    y = np.zeros(sr, dtype="float32")
    notes = transcribe_vocals(y, sr, fmin=80, fmax=1000)
    assert notes == []


@pytest.mark.slow
def test_real_vocal_stem_transcribes_to_lead_notes():
    if not CLIP.exists():
        pytest.skip(f"{CLIP} is missing; produced by the audio-analysis spike, not checked in")
    import soundfile as sf
    y, sr = sf.read(CLIP, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    notes = transcribe_vocals(y, sr, fmin=80, fmax=1000)
    assert notes
    assert all(n.role is Role.LEAD for n in notes)
    assert all(0 <= n.pitch <= 127 for n in notes)
