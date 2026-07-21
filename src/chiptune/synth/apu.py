"""Frame-clock render orchestration.

Walks each channel timeline one 60 Hz frame at a time, rendering that frame's
worth of samples from the appropriate oscillator and carrying oscillator phase
across frame boundaries so notes do not click.

Channel signals are produced in the 0-15 DAC range the NES mixer formulas expect.
Deliberate approximation: the real chip feeds those formulas integers, while we
feed continuous band-limited waveforms scaled into the same range. Quantizing to
integers would reintroduce exactly the aliasing that the band-limited pulse
oscillator exists to remove, and aliasing is far more audible than DAC
quantization. This is a considered trade.
"""
from __future__ import annotations

import numpy as np

from ..arrange.timeline import ChannelId, ChannelTimeline
from ..config import Config
from ..nes.tables import midi_to_hz, pulse_period, pulse_period_to_hz
from .noise import render_noise
from .pulse import PulseBank
from .triangle import render_triangle


def _frame_sample_bounds(frame: int, sample_rate: int, frame_rate: float) -> tuple[int, int]:
    """Exact sample span of a frame, computed from absolute edges so rounding never drifts."""
    start = int(round(frame * sample_rate / frame_rate))
    end = int(round((frame + 1) * sample_rate / frame_rate))
    return start, end


def _quantized_pulse_hz(pitch: int) -> float:
    """Route through the period register so pitch carries the chip's own quantization."""
    return pulse_period_to_hz(pulse_period(midi_to_hz(pitch)))


def render_channels(
    timelines: dict[ChannelId, ChannelTimeline],
    cfg: Config,
) -> dict[ChannelId, np.ndarray]:
    sr = cfg.sample_rate
    fr = cfg.frame_rate
    n_frames = max(len(t) for t in timelines.values())
    total = int(round(n_frames * sr / fr))

    bank = PulseBank(sample_rate=sr)
    out = {ch: np.zeros(total, dtype=np.float64) for ch in ChannelId}

    channel_cfg = {
        ChannelId.PULSE1: cfg.pulse1,
        ChannelId.PULSE2: cfg.pulse2,
        ChannelId.TRIANGLE: cfg.triangle,
        ChannelId.NOISE: cfg.noise,
    }

    phases = {ch: 0.0 for ch in ChannelId}
    noise_cursor = 0

    for ch in (ChannelId.PULSE1, ChannelId.PULSE2):
        duty = channel_cfg[ch].duty
        buf = out[ch]
        for f, ev in enumerate(timelines[ch].frames):
            a, b = _frame_sample_bounds(f, sr, fr)
            if b > total:
                b = total
            if a >= b:
                continue
            if ev.pitch is None:
                continue
            hz = _quantized_pulse_hz(ev.pitch)
            wave, phases[ch] = bank.render(hz, duty, b - a, phases[ch])
            buf[a:b] = wave * ev.volume        # 0-15 DAC range

    # Triangle: hardware has no volume control, so amplitude is fixed.
    tri_buf = out[ChannelId.TRIANGLE]
    tri_level = float(cfg.triangle.volume)
    for f, ev in enumerate(timelines[ChannelId.TRIANGLE].frames):
        a, b = _frame_sample_bounds(f, sr, fr)
        if b > total:
            b = total
        if a >= b or ev.pitch is None:
            continue
        wave, phases[ChannelId.TRIANGLE] = render_triangle(
            midi_to_hz(ev.pitch), b - a, sr, phases[ChannelId.TRIANGLE])
        tri_buf[a:b] = wave * tri_level

    noise_buf = out[ChannelId.NOISE]
    for f, ev in enumerate(timelines[ChannelId.NOISE].frames):
        a, b = _frame_sample_bounds(f, sr, fr)
        if b > total:
            b = total
        if a >= b or ev.percussion is None:
            continue
        voice = cfg.drums[ev.percussion.value]
        wave, noise_cursor = render_noise(
            voice.period_index, voice.mode, b - a, sr, noise_cursor)
        noise_buf[a:b] = wave * ev.volume

    # Per-channel trim from config, applied before the non-linear mixer.
    for ch in ChannelId:
        out[ch] *= cfg.levels.get(ch.value, 1.0)

    return out
