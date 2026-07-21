# tests/test_score.py
import numpy as np
import pytest
from chiptune.score import Role, Percussion, NoteEvent, TempoGrid, Score


def test_subdivision_times_are_evenly_spaced():
    grid = TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)
    # 120 bpm -> 0.5 s per beat; 1/16 notes -> 4 per beat -> 0.125 s apart
    times = grid.subdivision_times(subdivision=16, duration=1.0)
    np.testing.assert_allclose(times, [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])


def test_subdivision_times_respect_offset():
    grid = TempoGrid(bpm=120.0, offset=0.1, beats_per_bar=4)
    times = grid.subdivision_times(subdivision=16, duration=0.5)
    np.testing.assert_allclose(times, [0.1, 0.225, 0.35, 0.475])


def test_json_round_trip_preserves_everything():
    score = Score(
        tempo=TempoGrid(bpm=128.5, offset=0.02, beats_per_bar=4),
        notes=[
            NoteEvent(pitch=60, start=0.0, end=0.5, velocity=100, role=Role.LEAD),
            NoteEvent(pitch=36, start=0.0, end=0.1, velocity=127,
                      role=Role.PERCUSSION, percussion=Percussion.KICK),
        ],
        duration=2.0,
    )
    restored = Score.from_json(score.to_json())
    assert restored == score
    assert restored.notes[1].percussion is Percussion.KICK


def test_note_event_rejects_end_before_start():
    with pytest.raises(ValueError, match="end must be after start"):
        NoteEvent(pitch=60, start=1.0, end=0.5, velocity=100, role=Role.LEAD)


def test_percussion_role_requires_percussion_kind():
    with pytest.raises(ValueError, match="PERCUSSION notes require"):
        NoteEvent(pitch=36, start=0.0, end=0.1, velocity=100, role=Role.PERCUSSION)
