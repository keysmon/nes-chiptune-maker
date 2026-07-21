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
import sys

from ..config import Config
from ..nes.tables import playable_on_pulse, playable_on_triangle
from ..score import NoteEvent, Role, Score
from .arpeggio import arpeggiate
from .percussion import allocate_percussion
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
) -> list[NoteEvent | None]:
    """Reduce overlapping notes to one winning note per frame.

    Keeps the whole NoteEvent (not just the pitch) so callers can also read the
    winning note's velocity - the loudest/highest-priority pitch and the
    loudness that should drive the synth volume come from the same note.
    """
    chosen: list[NoteEvent | None] = [None] * n_frames
    for note in notes:
        for f in _frames_for(note, frame_rate):
            if f >= n_frames:
                break
            current = chosen[f]
            if current is None:
                chosen[f] = note
            elif pick_highest:
                chosen[f] = note if note.pitch > current.pitch else current
            else:
                chosen[f] = note if note.pitch < current.pitch else current
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


def _scale_by_velocity(base_volume: int, velocity: int, floor: float) -> int:
    """Scale a volume by note loudness, with `floor` keeping quiet notes audible.

    velocity=127 (max) always reproduces base_volume exactly, regardless of
    floor; velocity=0 gives base_volume * floor.
    """
    factor = floor + (1.0 - floor) * (velocity / 127.0)
    return int(round(base_volume * factor))


def _build_pitched_timeline(
    channel: ChannelId,
    pitches: list[int | None],
    cfg_channel,
    fixed_volume: bool,
    velocity_floor: float,
    velocities: list[int | None] | None = None,
) -> ChannelTimeline:
    frames: list[FrameEvent] = []
    held = 0
    prev: int | None = None
    for i, p in enumerate(pitches):
        if p is None:
            frames.append(SILENT)
            held = 0
            prev = None
            continue
        held = held + 1 if p == prev else 0
        if fixed_volume:
            # The triangle channel has no volume control; it is on or off.
            vol = cfg_channel.volume
        else:
            vol = _envelope_volume(cfg_channel, held, cfg_channel.volume)
            if velocities is not None:
                vol = _scale_by_velocity(vol, velocities[i], velocity_floor)
        frames.append(FrameEvent(pitch=p, volume=max(0, min(15, vol))))
        prev = p
    return ChannelTimeline(channel=channel, frames=frames)


def _warn_dropped(dropped: dict[str, list[int]]) -> None:
    """Report pitched notes dropped as unplayable, one line per allocate() call.

    Dropping an out-of-range pitch is the correct behaviour - clamping it to the
    register limit would sound a wrong note - but it must not be silent. This gives
    the pitched channels the same observability midi_io already gives dropped
    percussion, so a Phase-2 transcription feeding real pitches cannot lose notes
    without a trace.
    """
    parts = [
        f"{role} {len(pitches)} (e.g. MIDI {sorted(set(pitches))[0]})"
        for role, pitches in dropped.items()
        if pitches
    ]
    if not parts:
        return
    total = sum(len(p) for p in dropped.values())
    print(
        f"warning: dropped {total} unplayable pitched note(s) with no valid "
        f"11-bit period: {', '.join(parts)}",
        file=sys.stderr,
    )


def allocate(score: Score, cfg: Config) -> dict[ChannelId, ChannelTimeline]:
    fr = cfg.frame_rate
    n = frame_count(score.duration, fr)
    low, high = cfg.arrange.bass_low, cfg.arrange.bass_high

    # --- Pulse 1: lead, highest wins ---
    lead_notes = score.notes_with_role(Role.LEAD)
    lead = [nt for nt in lead_notes if playable_on_pulse(nt.pitch)]
    dropped_lead = [nt.pitch for nt in lead_notes if not playable_on_pulse(nt.pitch)]
    lead_winners = _monophonic(lead, n, fr, pick_highest=True)
    lead_pitches = [w.pitch if w is not None else None for w in lead_winners]
    lead_velocities = [w.velocity if w is not None else None for w in lead_winners]
    pulse1 = _build_pitched_timeline(
        ChannelId.PULSE1, lead_pitches, cfg.pulse1, fixed_volume=False,
        velocity_floor=cfg.arrange.velocity_floor, velocities=lead_velocities)

    # --- Triangle: bass, lowest wins, folded into range ---
    bass_notes = score.notes_with_role(Role.BASS)
    # Counted per note (like lead/harmony) rather than per reduced frame; folding
    # normally lands every bass pitch in a playable range, so this is a safety net.
    dropped_bass = [nt.pitch for nt in bass_notes
                    if not playable_on_triangle(fold_into_range(nt.pitch, low, high))]
    bass_winners = _monophonic(bass_notes, n, fr, pick_highest=False)
    bass_pitches = [w.pitch if w is not None else None for w in bass_winners]
    folded: list[int | None] = []
    for p in bass_pitches:
        if p is None:
            folded.append(None)
            continue
        q = fold_into_range(p, low, high)
        folded.append(q if playable_on_triangle(q) else None)
    # No `velocities` passed: fixed_volume=True means the triangle build never
    # reads them, so its "on or off" hardware constraint is untouched by velocity.
    triangle = _build_pitched_timeline(
        ChannelId.TRIANGLE, folded, cfg.triangle, fixed_volume=True,
        velocity_floor=cfg.arrange.velocity_floor)

    # --- Pulse 2: harmony, arpeggiated ---
    # Arpeggiation already re-strikes every chord tone every `arpeggio_frames`;
    # threading velocity through it would mean changing arpeggiate()'s return
    # type (and every existing caller of it), which is out of scope here. Pulse 2
    # keeps its pre-existing envelope-only volume.
    harmony_notes = score.notes_with_role(Role.HARMONY)
    harmony = [nt for nt in harmony_notes if playable_on_pulse(nt.pitch)]
    dropped_harmony = [nt.pitch for nt in harmony_notes if not playable_on_pulse(nt.pitch)]
    harmony_pitches = arpeggiate(harmony, n, fr, cfg.arrange.arpeggio_frames)
    pulse2 = _build_pitched_timeline(
        ChannelId.PULSE2, harmony_pitches, cfg.pulse2, fixed_volume=False,
        velocity_floor=cfg.arrange.velocity_floor)

    # --- Noise: percussion ---
    perc = score.notes_with_role(Role.PERCUSSION)
    noise = ChannelTimeline(
        channel=ChannelId.NOISE,
        frames=allocate_percussion(perc, n, fr, cfg.drums, velocity_floor=cfg.arrange.velocity_floor),
    )

    _warn_dropped({"lead": dropped_lead, "harmony": dropped_harmony, "bass": dropped_bass})

    return {
        ChannelId.PULSE1: pulse1,
        ChannelId.PULSE2: pulse2,
        ChannelId.TRIANGLE: triangle,
        ChannelId.NOISE: noise,
    }
