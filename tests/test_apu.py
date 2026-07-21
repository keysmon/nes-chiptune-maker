import numpy as np
import pytest

from chiptune.config import load_config
from chiptune.score import NoteEvent, Percussion, Role, Score, TempoGrid
from chiptune.arrange.allocator import allocate
from chiptune.arrange.timeline import ChannelId
from chiptune.synth.apu import render_channels


@pytest.fixture
def cfg():
    return load_config()


def render(score, cfg):
    return render_channels(allocate(score, cfg), cfg)


def test_all_channels_render_to_the_same_length(cfg):
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(72, 0.0, 1.0, 100, Role.LEAD)], duration=1.0)
    out = render(s, cfg)
    lengths = {len(v) for v in out.values()}
    assert len(lengths) == 1


def test_length_matches_duration(cfg):
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(72, 0.0, 1.0, 100, Role.LEAD)], duration=1.0)
    out = render(s, cfg)
    assert len(out[ChannelId.PULSE1]) == pytest.approx(cfg.sample_rate, abs=cfg.sample_rate // 50)


def test_silent_channel_is_actually_silent(cfg):
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(72, 0.0, 1.0, 100, Role.LEAD)], duration=1.0)
    out = render(s, cfg)
    assert np.abs(out[ChannelId.NOISE]).max() == 0.0


def test_lead_note_produces_energy_at_its_pitch(cfg):
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(69, 0.0, 1.0, 100, Role.LEAD)], duration=1.0)  # A4 = 440
    sig = render(s, cfg)[ChannelId.PULSE1]
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    peak = np.fft.rfftfreq(len(sig), 1 / cfg.sample_rate)[np.argmax(spec)]
    assert peak == pytest.approx(440.0, abs=5.0)


def test_rendering_is_deterministic(cfg):
    """Spec 6.1, scoped to Score -> wav."""
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(72, 0.0, 1.0, 100, Role.LEAD),
               NoteEvent(36, 0.0, 1.0, 100, Role.BASS)], duration=1.0)
    a = render(s, cfg)[ChannelId.PULSE1]
    b = render(s, cfg)[ChannelId.PULSE1]
    np.testing.assert_array_equal(a, b)


def test_percussion_renders_on_the_noise_channel(cfg):
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(36, 0.0, 0.05, 100, Role.PERCUSSION, percussion=Percussion.KICK)],
              duration=1.0)
    out = render(s, cfg)
    assert np.abs(out[ChannelId.NOISE]).max() > 0.0
