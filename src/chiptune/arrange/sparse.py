"""Sparse arrangement transforms: drop ornamental micro-notes and fuse
consecutive same-pitch notes into one sustained note.

The muddy-clutter problem these solve isn't only in HARMONY (see
`chiptune.arrange.chord_comp`): a note-by-note transcription of LEAD or BASS
also carries grace notes, slides, and re-quantization jitter that a real
arranger would gloss over into a single sustained pitch. `thin_melody` and
`simplify_bass` are the same pure NoteEvent-list transform applied to
different roles - two names because they're applied for different musical
reasons (LEAD ornaments vs. BASS walking noise), not because the logic
differs.
"""
from __future__ import annotations

from dataclasses import replace

from ..score import NoteEvent

# Consecutive same-pitch notes separated by less than this many seconds are
# treated as one held note, not two deliberate re-attacks - this absorbs
# transcription/quantization jitter, not a taste knob (the public interface
# here is (notes, min_seconds), so there's no config seam to thread a knob
# through even if it were one).
_MERGE_GAP_SECONDS = 0.02


def _thin(notes: list[NoteEvent], min_seconds: float) -> list[NoteEvent]:
    if not notes:
        return []

    ordered = sorted(notes, key=lambda n: n.start)
    long_enough = [n for n in ordered if n.duration >= min_seconds]
    if not long_enough:
        return []

    merged: list[NoteEvent] = [long_enough[0]]
    for n in long_enough[1:]:
        cur = merged[-1]
        if n.pitch == cur.pitch and n.start <= cur.end + _MERGE_GAP_SECONDS:
            # Extend, never shrink: an overlapping next note could otherwise
            # end earlier than the note it's merging into.
            merged[-1] = replace(cur, end=max(cur.end, n.end))
        else:
            merged.append(n)
    return merged


def thin_melody(notes: list[NoteEvent], min_seconds: float) -> list[NoteEvent]:
    """Drop LEAD notes shorter than `min_seconds`; fuse consecutive
    same-pitch notes into one sustained note."""
    return _thin(notes, min_seconds)


def simplify_bass(notes: list[NoteEvent], min_seconds: float) -> list[NoteEvent]:
    """Drop BASS notes shorter than `min_seconds`; fuse consecutive
    same-pitch notes into one sustained note."""
    return _thin(notes, min_seconds)
