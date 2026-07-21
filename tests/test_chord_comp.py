from chiptune.score import Role, TempoGrid
from chiptune.analysis.chords import ChordSegment
from chiptune.arrange.chord_comp import comp_chords


def _grid(bpm=120.0):
    return TempoGrid(bpm=bpm, offset=0.0, beats_per_bar=4)


def test_up_pattern_cycles_root_third_fifth():
    grid = _grid()
    seg = ChordSegment(start=0.0, end=grid.seconds_per_beat, root=0, is_minor=False)

    notes = comp_chords([seg], pattern="up", subdivision=3, octave=4, tones=3, grid=grid)

    assert [n.pitch for n in notes] == [60, 64, 67]
    assert all(n.role is Role.HARMONY for n in notes)
    assert all(seg.start <= n.start < n.end <= seg.end for n in notes)
    # monophonic: sorted by start, no overlap
    ordered = sorted(notes, key=lambda n: n.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start


def test_root_fifth_pattern_alternates_root_and_fifth():
    grid = _grid()
    seg = ChordSegment(start=0.0, end=grid.seconds_per_beat, root=0, is_minor=False)

    notes = comp_chords([seg], pattern="root_fifth", subdivision=3, octave=4, tones=3, grid=grid)

    assert set(n.pitch for n in notes) == {60, 67}


def test_updown_pattern_ascends_then_descends_without_doubling_endpoints():
    grid = _grid()
    seg = ChordSegment(start=0.0, end=grid.seconds_per_beat, root=0, is_minor=False)

    notes = comp_chords([seg], pattern="updown", subdivision=4, octave=4, tones=3, grid=grid)

    assert [n.pitch for n in notes] == [60, 64, 67, 64]


def test_minor_chord_uses_minor_third():
    grid = _grid()
    # A minor: root pitch class 9, is_minor True -> tones A(9) C(0) E(4) -> octave 4: 69, 72, 76
    seg = ChordSegment(start=0.0, end=grid.seconds_per_beat, root=9, is_minor=True)

    notes = comp_chords([seg], pattern="up", subdivision=3, octave=4, tones=3, grid=grid)

    assert [n.pitch for n in notes] == [69, 72, 76]


def test_drops_degenerate_tail_note_shorter_than_minimum():
    # bpm=120 -> seconds_per_beat=0.5; subdivision=1 -> step=0.5. A segment
    # 0.505s long (not a multiple of the step) leaves a 5ms tail note after
    # the one full-length note - that tail must be dropped, not emitted.
    grid = _grid()
    seg = ChordSegment(start=0.0, end=0.505, root=0, is_minor=False)

    notes = comp_chords([seg], pattern="up", subdivision=1, octave=4, tones=3, grid=grid)

    assert len(notes) == 1
    assert notes[0].start == 0.0
    assert notes[0].end == 0.5
    assert all(n.duration >= 0.01 for n in notes)


def test_high_octave_folds_pitches_into_valid_midi_range():
    """chord_octave is config/UI-exposed, so an extreme value must fold, not crash."""
    grid = _grid()
    seg = ChordSegment(start=0.0, end=grid.seconds_per_beat, root=8, is_minor=False)
    notes = comp_chords([seg], pattern="up", subdivision=3, octave=8, tones=4, grid=grid)
    assert notes, "should still emit notes"
    assert all(0 <= n.pitch <= 127 for n in notes), "all pitches must be valid MIDI"
