"""Role-locked voice allocation onto the four 2A03 channels.

Each source role permanently owns a channel:
    Pulse 1  <- lead      (monophonic, highest simultaneous pitch wins)
    Pulse 2  <- harmony   (arpeggiated; see arpeggio.py)
    Triangle <- bass      (lowest wins, octave-folded into range)
    Noise    <- drums

Reduction is the interesting part. When more notes want a channel than the
channel can play, the top voice is what the ear tracks as melody and the bottom
voice is what defines the harmony, so lead keeps the highest and bass keeps the
lowest. These are the choices a human 8-bit arranger makes.

Channel borrowing (spec 4.3, Phase 4) is deliberately not implemented here.
Strict role-locking is a subset of borrowing, so this code is not thrown away
when borrowing lands.
"""
from __future__ import annotations

import math

from ..config import Config
from ..nes.tables import playable_on_pulse, playable_on_triangle
from ..score import NoteEvent, Role, Score
from .arpeggio import arpeggiate
from .timeline import SILENT, ChannelId, ChannelTimeline, FrameEvent


def fold_into_range(pitch: int, low: int, high: int) -> int:
    """Shift `pitch` by whole octaves until it lands within [low, high]."""
    if low >= high:
        raise ValueError(f"low {low} must be below high {high}")
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def frame_count(duration: float, frame_rate: float) -> int:
    return max(1, int(math.ceil(duration * frame_rate)))


def _frames_for(note: NoteEvent, frame_rate: float) -> range:
    start = int(math.floor(note.start * frame_rate))
    end = int(math.ceil(note.end * frame_rate))
    return range(max(0, start), max(0, end))


def _monophonic(
    notes: list[NoteEvent],
    n_frames: int,
    frame_rate: float,
    pick_highest: bool,
) -> list[int | None]:
    """Reduce overlapping notes to one pitch per frame."""
    chosen: list[int | None] = [None] * n_frames
    for note in notes:
        for f in _frames_for(note, frame_rate):
            if f >= n_frames:
                break
            current = chosen[f]
            if current is None:
                chosen[f] = note.pitch
            elif pick_highest:
                chosen[f] = max(current, note.pitch)
            else:
                chosen[f] = min(current, note.pitch)
    return chosen


def _envelope_volume(cfg_channel, frames_held: int, base_volume: int) -> int:
    """Frame-quantized decay toward the sustain level."""
    if frames_held < cfg_channel.attack_frames:
        return base_volume
    decayed = frames_held - cfg_channel.attack_frames
    if cfg_channel.decay_frames <= 0 or decayed >= cfg_channel.decay_frames:
        return int(round(base_volume * cfg_channel.sustain))
    t = decayed / cfg_channel.decay_frames
    level = 1.0 + t * (cfg_channel.sustain - 1.0)
    return int(round(base_volume * level))


def _build_pitched_timeline(
    channel: ChannelId,
    pitches: list[int | None],
    cfg_channel,
    fixed_volume: bool,
) -> ChannelTimeline:
    frames: list[FrameEvent] = []
    held = 0
    prev: int | None = None
    for p in pitches:
        if p is None:
            frames.append(SILENT)
            held = 0
            prev = None
            continue
        held = held + 1 if p == prev else 0
        # The triangle channel has no volume control; it is on or off.
        vol = cfg_channel.volume if fixed_volume else _envelope_volume(
            cfg_channel, held, cfg_channel.volume)
        frames.append(FrameEvent(pitch=p, volume=max(0, min(15, vol))))
        prev = p
    return ChannelTimeline(channel=channel, frames=frames)


def allocate(score: Score, cfg: Config) -> dict[ChannelId, ChannelTimeline]:
    fr = cfg.frame_rate
    n = frame_count(score.duration, fr)

    # --- Pulse 1: lead, highest wins ---
    lead = [nt for nt in score.notes_with_role(Role.LEAD) if playable_on_pulse(nt.pitch)]
    lead_pitches = _monophonic(lead, n, fr, pick_highest=True)
    pulse1 = _build_pitched_timeline(ChannelId.PULSE1, lead_pitches, cfg.pulse1, fixed_volume=False)

    # --- Triangle: bass, lowest wins, folded into range ---
    bass_notes = score.notes_with_role(Role.BASS)
    bass_pitches = _monophonic(bass_notes, n, fr, pick_highest=False)
    folded: list[int | None] = []
    for p in bass_pitches:
        if p is None:
            folded.append(None)
            continue
        q = fold_into_range(p, cfg.arrange.bass_low, cfg.arrange.bass_high)
        folded.append(q if playable_on_triangle(q) else None)
    triangle = _build_pitched_timeline(
        ChannelId.TRIANGLE, folded, cfg.triangle, fixed_volume=True)

    # --- Pulse 2: harmony, arpeggiated ---
    harmony = [nt for nt in score.notes_with_role(Role.HARMONY)
               if playable_on_pulse(nt.pitch)]
    harmony_pitches = arpeggiate(harmony, n, fr, cfg.arrange.arpeggio_frames)
    pulse2 = _build_pitched_timeline(
        ChannelId.PULSE2, harmony_pitches, cfg.pulse2, fixed_volume=False)

    empty = lambda ch: ChannelTimeline(channel=ch, frames=[SILENT] * n)
    return {
        ChannelId.PULSE1: pulse1,
        ChannelId.PULSE2: pulse2,
        ChannelId.TRIANGLE: triangle,
        ChannelId.NOISE: empty(ChannelId.NOISE),     # Task 9
    }
