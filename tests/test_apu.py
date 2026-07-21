import numpy as np
import pytest

from chiptune.config import load_config
from chiptune.score import NoteEvent, Percussion, Role, Score, TempoGrid
from chiptune.arrange.allocator import allocate
from chiptune.arrange.timeline import SILENT, ChannelId, ChannelTimeline, FrameEvent
from chiptune.nes.tables import midi_to_hz, triangle_period, triangle_period_to_hz
from chiptune.synth.apu import render_channels


@pytest.fixture
def cfg():
    return load_config()


def render(score, cfg):
    return render_channels(allocate(score, cfg), cfg)


def _fft_peak_hz(sig, sample_rate):
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    return np.fft.rfftfreq(len(sig), 1 / sample_rate)[np.argmax(spec)]


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


def test_triangle_fundamental_is_period_quantized(cfg):
    """Wiring smoke: the triangle sounds at the period-quantized pitch, like the
    pulse channels. At bass range the quantization shift is sub-audible (<0.15 Hz),
    so this only confirms the triangle renders at the right note - it cannot by
    itself distinguish quantized from ideal; the high-note test below does that.
    """
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(48, 0.0, 1.0, 100, Role.BASS)], duration=1.0)  # C3
    sig = render(s, cfg)[ChannelId.TRIANGLE]
    quant = triangle_period_to_hz(triangle_period(midi_to_hz(48)))
    assert _fft_peak_hz(sig, cfg.sample_rate) == pytest.approx(quant, abs=2.0)


def test_triangle_renders_at_quantized_not_ideal_frequency(cfg):
    """Regression guard for the period-register quantization.

    At MIDI 84 the quantized frequency (~1055 Hz) differs from the ideal
    midi_to_hz (~1046 Hz) by ~9 Hz - resolvable by FFT, unlike at bass range. A
    hand-built triangle timeline isolates the render path so this fails if the
    triangle ever renders at the ideal frequency again instead of the quantized one.
    """
    n_frames = 90  # 1.5 s at 60 fps -> ~0.67 Hz FFT bins
    timelines = {ch: ChannelTimeline(ch, [SILENT] * n_frames) for ch in ChannelId}
    timelines[ChannelId.TRIANGLE] = ChannelTimeline(
        ChannelId.TRIANGLE, [FrameEvent(pitch=84, volume=15)] * n_frames)

    sig = render_channels(timelines, cfg)[ChannelId.TRIANGLE]
    peak = _fft_peak_hz(sig, cfg.sample_rate)

    quant = triangle_period_to_hz(triangle_period(midi_to_hz(84)))
    ideal = midi_to_hz(84)
    assert peak == pytest.approx(quant, abs=3.0)
    assert abs(peak - ideal) > 5.0, "triangle must render the quantized pitch, not the ideal one"
