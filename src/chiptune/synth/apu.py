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
from scipy.signal import butter, sosfiltfilt

from ..arrange.timeline import ChannelId, ChannelTimeline
from ..config import Config, VibratoConfig
from ..nes.tables import (
    midi_to_hz,
    pulse_period,
    pulse_period_to_hz,
    triangle_period,
    triangle_period_to_hz,
)
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


def _quantized_triangle_hz(pitch: int) -> float:
    """Triangle counterpart of _quantized_pulse_hz: the triangle has its own /32
    period register, so its pitch must carry the same chip quantization the pulses do
    rather than sound at the ideal frequency."""
    return triangle_period_to_hz(triangle_period(midi_to_hz(pitch)))


def _vibrato_multiplier(note_age_frames: int, frame_rate: float, vib: VibratoConfig) -> float:
    """Sinusoidal pitch LFO multiplier for the lead voice.

    `note_age_frames` counts frames since the current pitch was struck (0 on the
    onset frame), so the LFO phase is derived from absolute time since the delay
    elapsed rather than reset each frame - that is what keeps it phase-continuous
    across frame boundaries instead of clicking back to zero every render call.
    """
    if not vib.enabled or note_age_frames < vib.delay_frames:
        return 1.0
    t = (note_age_frames - vib.delay_frames) / frame_rate
    lfo = np.sin(2.0 * np.pi * vib.rate_hz * t)
    return 2.0 ** (vib.depth_semitones * lfo / 12.0)


# A drum hit is not a flat gate of noise: real percussion is a fast attack followed
# by a decay. Rendering the noise burst at constant amplitude (as a raw rectangular
# window) sounds abrupt and hissy - the hard edges click and the un-decayed noise
# reads as a sustained blast rather than a thump/crack. This envelope shapes each hit.
_PERC_DECAY_SHAPE = 4.0     # exp decay reaches e^-4 (~1.8%) by the end of the hit
_PERC_ATTACK_S = 0.002      # onset fade, kills the leading click
_PERC_RELEASE_S = 0.004     # tail fade, kills the trailing click


def _percussion_envelope(n: int, sr: int) -> np.ndarray:
    """Amplitude envelope for one drum hit spanning `n` samples: exponential decay
    (proportional to hit length, so `frames` controls decay time) plus short attack
    and release fades that remove the rectangular-window clicks."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    env = np.exp(-_PERC_DECAY_SHAPE * np.arange(n) / n)
    a = min(max(1, int(_PERC_ATTACK_S * sr)), n)
    env[:a] *= np.linspace(0.0, 1.0, a)
    r = min(max(1, int(_PERC_RELEASE_S * sr)), n)
    env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


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
        prev_pitch: int | None = None
        note_age = 0  # frames the current pitch has been held; reset on change/silence
        for f, ev in enumerate(timelines[ch].frames):
            a, b = _frame_sample_bounds(f, sr, fr)
            if b > total:
                b = total
            if a >= b:
                continue
            if ev.pitch is None:
                prev_pitch = None
                note_age = 0
                continue
            if ev.pitch == prev_pitch:
                note_age += 1
            else:
                prev_pitch = ev.pitch
                note_age = 0
            hz = _quantized_pulse_hz(ev.pitch)
            if ch is ChannelId.PULSE1:
                # Lead-only vibrato. Pulse 2 is arpeggiated - modulating it on top
                # of the arpeggio would read as detuned rather than expressive.
                hz *= _vibrato_multiplier(note_age, fr, cfg.vibrato)
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
            _quantized_triangle_hz(ev.pitch), b - a, sr, phases[ChannelId.TRIANGLE])
        tri_buf[a:b] = wave * tri_level

    # Render each drum hit as one contiguous, enveloped noise burst rather than a
    # sequence of independent flat frames. A hit is a run of consecutive frames with
    # the same percussion kind and volume; the whole run gets a single attack-decay
    # envelope so it sounds percussive instead of like a gated blast of white noise.
    noise_buf = out[ChannelId.NOISE]
    noise_frames = timelines[ChannelId.NOISE].frames
    nf = len(noise_frames)
    f = 0
    while f < nf:
        ev = noise_frames[f]
        if ev.percussion is None:
            f += 1
            continue
        kind, vol = ev.percussion, ev.volume
        start_f = f
        f += 1
        while f < nf and noise_frames[f].percussion is kind and noise_frames[f].volume == vol:
            f += 1
        a, _ = _frame_sample_bounds(start_f, sr, fr)
        _, b = _frame_sample_bounds(f - 1, sr, fr)
        b = min(b, total)
        if a >= b:
            continue
        voice = cfg.drums[kind.value]
        wave, noise_cursor = render_noise(voice.period_index, voice.mode, b - a, sr, noise_cursor)
        noise_buf[a:b] = wave * vol * _percussion_envelope(b - a, sr)

    # Low-pass the whole noise channel to tame the harsh, hissy top end of white noise.
    # Real console output was filtered too; raw LFSR noise is brighter than a NES ever
    # sounded. Zero-phase (filtfilt) so drum transients keep their timing.
    if cfg.noise_lowpass_hz and 0 < cfg.noise_lowpass_hz < sr / 2 and noise_buf.any():
        sos = butter(2, cfg.noise_lowpass_hz, btype="low", fs=sr, output="sos")
        noise_buf[:] = sosfiltfilt(sos, noise_buf)

    # Per-channel trim from config, applied before the non-linear mixer.
    for ch in ChannelId:
        out[ch] *= cfg.levels.get(ch.value, 1.0)

    return out
