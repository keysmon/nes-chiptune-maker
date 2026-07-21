from pathlib import Path

import numpy as np
import pytest

from chiptune.analysis import vocals as V
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


def test_voiced_frame_with_nonfinite_f0_does_not_emit_pitch_zero_note(monkeypatch):
    # pyin can (in principle, on a future version) flag a frame voiced=True
    # while f0 is NaN. np.round(...).astype(int64) casts that NaN to 0 on this
    # platform, which would otherwise pass through as a bogus MIDI-0 LEAD note.
    sr = 22050
    hop = V.HOP_LENGTH
    n_frames = 20

    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    vprob = np.zeros(n_frames)

    # A legit voiced run (A3 = 220Hz -> MIDI 57), frames 0-9.
    f0[0:10] = 220.0
    voiced[0:10] = True
    vprob[0:10] = 0.9

    # frames 10-11 unvoiced (silence gap), then a bogus voiced-but-NaN frame.
    voiced[12] = True
    f0[12] = np.nan
    vprob[12] = 0.9

    monkeypatch.setattr(V.librosa, "pyin", lambda *a, **k: (f0, voiced, vprob))

    notes = transcribe_vocals(np.zeros(n_frames * hop, dtype="float32"), sr, fmin=80, fmax=1000, min_duration=0.0)
    assert all(n.pitch != 0 for n in notes)
    assert any(abs(n.pitch - 57) <= 1 for n in notes)  # the legit A3 run still comes through


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
