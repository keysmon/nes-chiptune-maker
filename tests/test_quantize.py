# tests/test_quantize.py
import pytest
from chiptune.score import Role, NoteEvent, TempoGrid, Score
from chiptune.quantize import quantize_score


def make_score(notes, bpm=120.0):
    return Score(tempo=TempoGrid(bpm=bpm, offset=0.0, beats_per_bar=4),
                 notes=notes, duration=4.0)


def test_full_strength_snaps_onset_to_nearest_subdivision():
    # 120 bpm, 1/16 grid -> 0.125 s steps. 0.13 snaps to 0.125.
    s = make_score([NoteEvent(60, 0.13, 0.4, 100, Role.LEAD)])
    out = quantize_score(s, subdivision=16, strength=1.0, min_duration=0.01)
    assert out.notes[0].start == pytest.approx(0.125)


def test_zero_strength_is_a_no_op():
    s = make_score([NoteEvent(60, 0.13, 0.4, 100, Role.LEAD)])
    out = quantize_score(s, subdivision=16, strength=0.0, min_duration=0.01)
    assert out.notes[0].start == pytest.approx(0.13)


def test_partial_strength_blends_toward_the_grid():
    s = make_score([NoteEvent(60, 0.145, 0.4, 100, Role.LEAD)])
    out = quantize_score(s, subdivision=16, strength=0.5, min_duration=0.01)
    # halfway between 0.145 and 0.125
    assert out.notes[0].start == pytest.approx(0.135)


def test_quantization_never_produces_zero_length_notes():
    # onset and offset would both snap to 0.125
    s = make_score([NoteEvent(60, 0.124, 0.126, 100, Role.LEAD)])
    out = quantize_score(s, subdivision=16, strength=1.0, min_duration=0.02)
    assert out.notes[0].duration >= 0.02


def test_pitch_is_coerced_to_integer_semitone():
    s = make_score([NoteEvent(60, 0.0, 0.5, 100, Role.LEAD)])
    out = quantize_score(s, subdivision=16, strength=1.0, min_duration=0.01)
    assert isinstance(out.notes[0].pitch, int)


def test_notes_stay_sorted_by_onset():
    s = make_score([
        NoteEvent(60, 0.51, 0.9, 100, Role.LEAD),
        NoteEvent(62, 0.13, 0.4, 100, Role.LEAD),
    ])
    out = quantize_score(s, subdivision=16, strength=1.0, min_duration=0.01)
    assert [n.start for n in out.notes] == sorted(n.start for n in out.notes)
