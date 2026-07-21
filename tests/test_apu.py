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


def test_percussion_hit_decays_and_has_soft_edges(cfg):
    """A drum hit must be an attack-decay transient, not a flat gated noise burst:
    it should decay over its length and fade in/out to avoid onset/offset clicks."""
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(38, 0.0, 0.1, 120, Role.PERCUSSION, percussion=Percussion.SNARE)],
              duration=1.0)
    out = render(s, cfg)[ChannelId.NOISE]
    nz = np.nonzero(np.abs(out) > 1e-6)[0]
    assert len(nz) > 0, "the hit produced no sound"
    region = out[nz[0]:nz[-1] + 1]
    q = len(region) // 4
    early = np.abs(region[:q]).max()
    late = np.abs(region[-q:]).max()
    assert late < 0.5 * early, f"hit did not decay (early {early:.3f} vs late {late:.3f}) - still a flat blast"
    # soft edges: the very first and last samples are near-silent (fades kill clicks)
    assert abs(region[0]) < 0.3 * early
    assert abs(region[-1]) < 0.3 * early


def test_noise_lowpass_reduces_high_frequency_energy(cfg):
    """The noise low-pass must actually roll off the harsh top end of the drums."""
    import dataclasses
    s = Score(TempoGrid(120.0, 0.0, 4),
              [NoteEvent(38, t / 10, t / 10 + 0.08, 120, Role.PERCUSSION, percussion=Percussion.SNARE)
               for t in range(8)], duration=1.0)

    def hi_frac(out):
        spec = np.abs(np.fft.rfft(out)) ** 2
        freqs = np.fft.rfftfreq(len(out), 1 / cfg.sample_rate)
        return spec[freqs > 6000].sum() / max(spec.sum(), 1e-12)

    filtered = render(s, cfg)[ChannelId.NOISE]
    unfiltered = render(s, dataclasses.replace(cfg, noise_lowpass_hz=0.0))[ChannelId.NOISE]
    assert hi_frac(filtered) < 0.5 * hi_frac(unfiltered), "low-pass did not reduce high-frequency energy"


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


def test_vibrato_modulates_a_sustained_lead(cfg):
    # a long held note should show sidebands / frequency spread vs a vibrato-off render
    import dataclasses
    s = Score(TempoGrid(120., 0., 4), [NoteEvent(69, 0.0, 2.0, 100, Role.LEAD)], 2.0)  # A4 held 2s
    on = render(s, cfg)[ChannelId.PULSE1]
    off = render(s, dataclasses.replace(cfg, vibrato=dataclasses.replace(cfg.vibrato, enabled=False)))[ChannelId.PULSE1]

    def spread(x):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / cfg.sample_rate)
        peak = f[np.argmax(spec)]
        band = (f > peak * 0.98) & (f < peak * 1.02)
        return spec[band].sum() / spec.max()

    assert spread(on) > spread(off), "vibrato should widen the fundamental"


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
