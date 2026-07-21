import pytest

from chiptune.analysis.density import score_density
from chiptune.score import NoteEvent, Role, Score, TempoGrid


def _score(notes, duration):
    return Score(tempo=TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4), notes=notes, duration=duration)


def test_sustained_lead_note_has_near_full_active_fraction():
    notes = [NoteEvent(60, 0.0, 2.0, 80, Role.LEAD)]
    score = _score(notes, duration=2.0)

    density = score_density(score, frame_rate=60.0)

    assert density["per_role_active"][Role.LEAD] == pytest.approx(1.0)
    assert density["per_role_active"][Role.HARMONY] == 0.0
    assert density["mean_simultaneous"] == pytest.approx(1.0)
    assert density["notes_per_second"][Role.LEAD] == pytest.approx(0.5)  # 1 note / 2.0s


def test_overlapping_harmony_notes_raise_mean_simultaneous():
    lead = [NoteEvent(60, 0.0, 2.0, 80, Role.LEAD)]
    lead_only = score_density(_score(lead, duration=2.0), frame_rate=60.0)

    harmony = [
        NoteEvent(64, 0.0, 2.0, 70, Role.HARMONY),
        NoteEvent(67, 0.0, 2.0, 70, Role.HARMONY),
        NoteEvent(71, 0.0, 2.0, 70, Role.HARMONY),
    ]
    with_harmony = score_density(_score(lead + harmony, duration=2.0), frame_rate=60.0)

    assert with_harmony["mean_simultaneous"] > lead_only["mean_simultaneous"]
    assert with_harmony["mean_simultaneous"] == pytest.approx(4.0)
    assert with_harmony["notes_per_second"][Role.HARMONY] == pytest.approx(1.5)  # 3 / 2.0s


def test_empty_score_returns_zeros_without_crashing():
    density = score_density(_score([], duration=0.0), frame_rate=60.0)

    assert density["mean_simultaneous"] == 0.0
    assert all(v == 0.0 for v in density["per_role_active"].values())
    assert all(v == 0.0 for v in density["notes_per_second"].values())
