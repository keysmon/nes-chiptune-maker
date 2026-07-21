import pytest

from chiptune.config import load_config
from chiptune.score import NoteEvent, Percussion, Role, Score, TempoGrid
from chiptune.arrange.percussion import allocate_percussion
from chiptune.arrange.allocator import allocate
from chiptune.arrange.timeline import ChannelId

PRIORITY_CASES = [
    ([Percussion.KICK, Percussion.SNARE, Percussion.HAT], Percussion.KICK),
    ([Percussion.SNARE, Percussion.HAT], Percussion.SNARE),
    ([Percussion.HAT], Percussion.HAT),
]


def hit(kind, t, pitch=36):
    return NoteEvent(pitch, t, t + 0.05, 100, Role.PERCUSSION, percussion=kind)


@pytest.fixture
def drums():
    return load_config().drums


def test_a_hit_sounds_for_its_configured_frame_count(drums):
    frames = allocate_percussion([hit(Percussion.KICK, 0.0)], 60, 60.0, drums)
    kick_frames = drums["kick"].frames
    assert all(f.percussion is Percussion.KICK for f in frames[:kick_frames])
    assert frames[kick_frames].percussion is None


@pytest.mark.parametrize("kinds,winner", PRIORITY_CASES)
def test_collisions_resolve_by_priority(kinds, winner, drums):
    frames = allocate_percussion([hit(k, 0.0) for k in kinds], 60, 60.0, drums)
    assert frames[0].percussion is winner


def test_hits_at_different_times_all_sound(drums):
    frames = allocate_percussion(
        [hit(Percussion.KICK, 0.0), hit(Percussion.SNARE, 0.5)], 60, 60.0, drums)
    assert frames[0].percussion is Percussion.KICK
    assert frames[30].percussion is Percussion.SNARE


def test_volume_comes_from_config(drums):
    frames = allocate_percussion([hit(Percussion.KICK, 0.0)], 60, 60.0, drums)
    assert frames[0].volume == drums["kick"].volume


def test_allocator_now_fills_noise():
    cfg = load_config()
    s = Score(tempo=TempoGrid(120.0, 0.0, 4),
              notes=[hit(Percussion.KICK, 0.0), hit(Percussion.SNARE, 0.5)],
              duration=1.0)
    tl = allocate(s, cfg)[ChannelId.NOISE]
    kinds = {f.percussion for f in tl.frames if f.percussion is not None}
    assert kinds == {Percussion.KICK, Percussion.SNARE}
