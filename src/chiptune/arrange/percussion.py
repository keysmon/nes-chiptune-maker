"""Map drum hits onto the single noise channel.

Only one hit can sound at a time. When hits collide, low-frequency hits carry
the groove: a lost hi-hat is barely noticed, a lost kick is. Hence kick > snare > hat.
"""
from __future__ import annotations

import math

from ..config import DrumVoice
from ..score import NoteEvent, Percussion
from .timeline import SILENT, FrameEvent

PRIORITY: dict[Percussion, int] = {
    Percussion.KICK: 3,
    Percussion.SNARE: 2,
    Percussion.HAT: 1,
}


def allocate_percussion(
    notes: list[NoteEvent],
    n_frames: int,
    frame_rate: float,
    drums: dict[str, DrumVoice],
) -> list[FrameEvent]:
    missing = {n.percussion.value for n in notes if n.percussion} - set(drums)
    if missing:
        raise ValueError(f"config has no [drums.{sorted(missing)[0]}] section")

    frames: list[FrameEvent] = [SILENT] * n_frames
    winners: list[Percussion | None] = [None] * n_frames

    for note in sorted(notes, key=lambda n: n.start):
        if note.percussion is None:
            raise ValueError("percussion notes must carry a `percussion` kind")
        voice = drums[note.percussion.value]
        start = int(math.floor(note.start * frame_rate))
        for i in range(voice.frames):
            f = start + i
            if not 0 <= f < n_frames:
                continue
            existing = winners[f]
            if existing is None or PRIORITY[note.percussion] > PRIORITY[existing]:
                winners[f] = note.percussion
                frames[f] = FrameEvent(
                    pitch=None, volume=voice.volume, percussion=note.percussion)

    return frames
