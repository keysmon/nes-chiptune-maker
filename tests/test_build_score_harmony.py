from types import SimpleNamespace
from chiptune.score import NoteEvent, Role, TempoGrid
from chiptune.analysis.chords import ChordSegment
from chiptune.analysis.build_score import _build_chords_harmony
from chiptune.arrange.chord_comp import comp_chords


def _grid():
    return TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)


def _arr(**over):
    base = dict(harmony_source="select", chord_comp_pattern="up", chord_subdivision=3,
                chord_octave=4, chord_tones=3, select_min_gap=0.10,
                harmony_rest_on_busy_melody=False)
    base.update(over)
    return SimpleNamespace(**base)


def test_arp_source_reproduces_comp_chords_exactly():
    grid = _grid()
    chords = [ChordSegment(start=0.0, end=grid.seconds_per_beat, root=0, is_minor=False)]
    expected = comp_chords(chords, pattern="up", subdivision=3, octave=4, tones=3, grid=grid)
    got = _build_chords_harmony(_arr(harmony_source="arp"), None, chords, [], grid)
    # Full NoteEvent equality (pitch, start, end, velocity, role, percussion) -
    # the global constraint is that arp reproduces today's comp_chords output
    # EXACTLY, not just its timing/pitch.
    assert got == expected


def test_select_source_uses_selection_and_is_harmony_monophonic():
    grid = _grid()
    chords = [ChordSegment(start=0.0, end=2.0, root=0, is_minor=False)]
    cands = [NoteEvent(pitch=62, start=0.1, end=0.9, velocity=110, role=Role.HARMONY)]
    got = _build_chords_harmony(_arr(harmony_source="select"), cands, chords, [], grid)
    assert got and all(n.role is Role.HARMONY for n in got)
    assert all(n.pitch % 12 in {0, 4, 7} for n in got)
    for a, b in zip(got, got[1:]):
        assert a.end <= b.start
