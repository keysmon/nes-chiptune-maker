import pytest

from chiptune.config import load_config
from chiptune.score import NoteEvent, Role, Score, TempoGrid
from chiptune.arrange.arpeggio import arpeggiate
from chiptune.arrange.allocator import allocate
from chiptune.arrange.timeline import ChannelId


def notes(pitches, start=0.0, end=1.0):
    return [NoteEvent(p, start, end, 90, Role.HARMONY) for p in pitches]


def test_single_note_does_not_cycle():
    out = arpeggiate(notes([60]), n_frames=10, frame_rate=60.0, arpeggio_frames=2)
    assert out == [60] * 10


def test_triad_cycles_through_all_chord_tones():
    out = arpeggiate(notes([60, 64, 67]), n_frames=12, frame_rate=60.0, arpeggio_frames=1)
    assert out[:6] == [60, 64, 67, 60, 64, 67]


def test_arpeggio_frames_controls_dwell_time():
    out = arpeggiate(notes([60, 64, 67]), n_frames=12, frame_rate=60.0, arpeggio_frames=2)
    assert out[:6] == [60, 60, 64, 64, 67, 67]


def test_five_note_cluster_still_visits_every_tone():
    """The over-budget case from the test asset."""
    out = arpeggiate(notes([60, 64, 67, 71, 74]), n_frames=60,
                     frame_rate=60.0, arpeggio_frames=1)
    assert set(out[:5]) == {60, 64, 67, 71, 74}


def test_silence_where_no_note_sounds():
    out = arpeggiate(notes([60], start=0.5, end=1.0), n_frames=60,
                     frame_rate=60.0, arpeggio_frames=2)
    assert out[0] is None
    assert out[40] == 60


def test_exactly_one_pitch_per_frame():
    out = arpeggiate(notes([60, 64, 67, 71, 74]), n_frames=30,
                     frame_rate=60.0, arpeggio_frames=2)
    assert all(v is None or isinstance(v, int) for v in out)
    assert len(out) == 30


def test_allocator_now_fills_pulse2():
    cfg = load_config()
    s = Score(tempo=TempoGrid(120.0, 0.0, 4),
              notes=notes([60, 64, 67]), duration=1.0)
    tl = allocate(s, cfg)[ChannelId.PULSE2]
    sounded = {f.pitch for f in tl.frames if f.pitch is not None}
    assert sounded == {60, 64, 67}
