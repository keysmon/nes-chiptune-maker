from chiptune.arrange.sparse import simplify_bass, thin_melody
from chiptune.score import NoteEvent, Role


def _n(pitch, start, end, role):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=80, role=role)


def test_thin_melody_drops_short_ornament_note_between_two_longer_notes():
    notes = [
        _n(60, 0.0, 0.5, Role.LEAD),
        _n(65, 0.5, 0.51, Role.LEAD),  # 10ms grace-note ornament
        _n(67, 0.52, 1.0, Role.LEAD),
    ]

    out = thin_melody(notes, min_seconds=0.05)

    assert [n.pitch for n in out] == [60, 67]


def test_thin_melody_merges_consecutive_same_pitch_notes():
    notes = [_n(60, 0.0, 0.5, Role.LEAD), _n(60, 0.5, 1.0, Role.LEAD)]

    out = thin_melody(notes, min_seconds=0.05)

    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 1.0


def test_thin_melody_handles_empty_list():
    assert thin_melody([], min_seconds=0.05) == []


def test_simplify_bass_drops_short_note_between_two_longer_notes():
    notes = [
        _n(30, 0.0, 0.5, Role.BASS),
        _n(35, 0.5, 0.51, Role.BASS),  # 10ms ornament
        _n(33, 0.52, 1.0, Role.BASS),
    ]

    out = simplify_bass(notes, min_seconds=0.05)

    assert [n.pitch for n in out] == [30, 33]


def test_simplify_bass_merges_consecutive_same_pitch_notes():
    notes = [_n(30, 0.0, 0.5, Role.BASS), _n(30, 0.5, 1.0, Role.BASS)]

    out = simplify_bass(notes, min_seconds=0.05)

    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 1.0


def test_simplify_bass_handles_empty_list():
    assert simplify_bass([], min_seconds=0.05) == []
