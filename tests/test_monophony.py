"""Unit tests for the shared strict-monophony helper."""
from chiptune.arrange.monophony import enforce_monophonic
from chiptune.score import NoteEvent, Role


def _n(pitch, start, end, vel=80, role=Role.HARMONY):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=vel, role=role)


def test_empty_returns_empty():
    assert enforce_monophonic([]) == []


def test_non_overlapping_notes_pass_through():
    notes = [_n(60, 0.0, 0.5), _n(64, 0.5, 1.0)]
    assert enforce_monophonic(notes) == notes


def test_overlap_is_clamped_to_next_start():
    out = enforce_monophonic([_n(60, 0.0, 0.8), _n(64, 0.5, 1.0)])
    assert out[0].end == 0.5 and out[1].start == 0.5
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start


def test_zero_length_after_clamp_is_dropped():
    # two notes starting at the same time: the earlier-listed clamps to zero length -> dropped
    out = enforce_monophonic([_n(60, 0.0, 1.0), _n(64, 0.0, 0.5)])
    assert len(out) == 1


def test_output_is_sorted_and_harmony_role():
    out = enforce_monophonic([_n(64, 0.5, 1.0, role=Role.LEAD), _n(60, 0.0, 0.4, role=Role.LEAD)])
    assert [n.start for n in out] == [0.0, 0.5]
    assert all(n.role is Role.HARMONY for n in out)
