from chiptune.score import NoteEvent, Role
from chiptune.arrange.rearticulate import rearticulate


def test_adjacent_same_pitch_notes_get_a_gap():
    notes = [NoteEvent(60, 0.0, 0.5, 100, Role.LEAD), NoteEvent(60, 0.5, 1.0, 100, Role.LEAD)]
    out = rearticulate(notes, gap_seconds=0.02)
    assert out[0].end <= 0.5 - 0.02 + 1e-9, "a gap must precede the repeated note"


def test_different_pitch_notes_untouched():
    notes = [NoteEvent(60, 0.0, 0.5, 100, Role.LEAD), NoteEvent(62, 0.5, 1.0, 100, Role.LEAD)]
    out = rearticulate(notes, gap_seconds=0.02)
    assert out[0].end == 0.5


def test_gap_zero_is_a_noop():
    notes = [NoteEvent(60, 0.0, 0.5, 100, Role.LEAD), NoteEvent(60, 0.5, 1.0, 100, Role.LEAD)]
    assert [n.end for n in rearticulate(notes, 0.0)] == [0.5, 1.0]
