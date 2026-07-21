"""Re-attack repeated same-pitch notes.

The per-frame pitch model (see allocator._monophonic) can only tell "sounding"
from "silent"; it has no way to distinguish one held note from two consecutive
identical notes with no gap between them, so they fuse into a single sustained
tone. Trimming a hair off the end of the first note gives the ear - and the
envelope's attack_frames - a silence to re-trigger on.
"""
from __future__ import annotations

from dataclasses import replace

from ..score import NoteEvent


def rearticulate(
    notes: list[NoteEvent],
    gap_seconds: float,
    min_duration: float = 0.03,
) -> list[NoteEvent]:
    """Shorten each note that is immediately followed by a same-pitch, same-role
    note so a `gap_seconds` silence separates the two. `gap_seconds` of 0 disables
    this (a no-op copy is returned). A trim that would shrink a note below
    `min_duration` is dropped rather than applied.
    """
    if gap_seconds <= 0:
        return list(notes)

    out = list(notes)
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        if cur.pitch != nxt.pitch or cur.role != nxt.role:
            continue
        if nxt.start - cur.end >= gap_seconds:
            continue
        new_end = nxt.start - gap_seconds
        # Drop the trim if it would leave the note below min_duration, and never
        # let it reach or cross the onset (guards NoteEvent's end>start even if a
        # caller passes min_duration=0).
        if new_end <= cur.start or new_end - cur.start < min_duration:
            continue
        out[i] = replace(cur, end=new_end)
    return out
