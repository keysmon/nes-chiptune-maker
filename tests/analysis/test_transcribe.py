from pathlib import Path

import numpy as np
import pytest

from chiptune.score import Role
from chiptune.analysis import transcribe as T

CLIP = Path("out/spike_demucs/htdemucs/pop_src/bass.wav")


def test_maps_basic_pitch_output_to_note_events(monkeypatch):
    fake = [(0.0, 0.5, 60, 0.8, []), (0.5, 0.52, 62, 0.5, [])]  # 2nd too short
    monkeypatch.setattr(T, "_run_basic_pitch", lambda wav: fake)
    notes = T.transcribe_pitched(np.zeros(1000, dtype="float32"), 22050, Role.BASS, min_duration=0.05)
    assert len(notes) == 1
    assert notes[0].pitch == 60 and notes[0].role is Role.BASS
    assert 1 <= notes[0].velocity <= 127


def test_zero_duration_and_out_of_range_pitch_do_not_crash(monkeypatch):
    fake = [
        (0.0, 0.5, 60, 0.8, []),    # normal note
        (0.5, 0.5, 61, 0.5, []),    # degenerate: end == start, slips past `< min_duration` when it's 0.0
        (0.6, 0.8, 200, 0.5, []),   # out-of-range pitch; basic-pitch shouldn't emit this, but don't crash
    ]
    monkeypatch.setattr(T, "_run_basic_pitch", lambda wav: fake)
    notes = T.transcribe_pitched(np.zeros(1000, dtype="float32"), 22050, Role.BASS, min_duration=0.0)
    assert all(0 <= n.pitch <= 127 for n in notes)
    assert all(n.end > n.start for n in notes)
    assert len(notes) == 2  # the zero-duration tuple is dropped; the other two survive (pitch clamped)
    assert notes[1].pitch == 127  # 200 clamped down to the MIDI ceiling


@pytest.mark.slow
def test_slow_real_bass_stem_transcribes_to_bass_range_notes():
    if not CLIP.exists():
        pytest.skip(f"{CLIP} is missing; produced by the audio-analysis spike, not checked in")
    import soundfile as sf
    y, sr = sf.read(CLIP, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    notes = T.transcribe_pitched(y, sr, Role.BASS, min_duration=0.05)
    assert notes
    assert all(n.role is Role.BASS for n in notes)
    assert all(0 <= n.pitch <= 127 for n in notes)
