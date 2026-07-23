"""Strict monophony for a HARMONY note list.

Shared by the harmony comps (comp_select, phantom_echo): both feed the
allocator's arpeggiate step, which buzz-cycles any overlapping HARMONY pitches,
so each must hand it a strictly-monophonic list.
"""
from __future__ import annotations

from ..score import NoteEvent, Role


def enforce_monophonic(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Sort by start and clamp each note's end to the next note's start, dropping
    any note that collapses to zero length. Guarantees strict monophony (no two
    notes overlap). Output notes are tagged Role.HARMONY."""
    ordered = sorted(notes, key=lambda n: n.start)
    out: list[NoteEvent] = []
    for i, n in enumerate(ordered):
        end = n.end
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1].start)
        if end > n.start:
            out.append(NoteEvent(pitch=n.pitch, start=n.start, end=end,
                                 velocity=n.velocity, role=Role.HARMONY))
    return out
