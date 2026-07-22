"""Note-selection harmony comp: keep the important real harmony notes, pruned
to a sparsity budget and snapped to the detected chords, instead of a fixed
arpeggio. Design: docs/superpowers/specs/2026-07-22-note-selection-comp-design.md
"""
from __future__ import annotations

from ..score import NoteEvent, Role, TempoGrid
from ..analysis.chords import ChordSegment

_HARMONY_VELOCITY = 80


def _tone_pcs(chord: ChordSegment) -> set[int]:
    """The three chord-tone pitch classes (0-11) of `chord`."""
    third = 3 if chord.is_minor else 4
    return {chord.root % 12, (chord.root + third) % 12, (chord.root + 7) % 12}


def _chord_at(chords: list[ChordSegment], t: float) -> ChordSegment | None:
    """The chord segment containing time `t`, or None if `t` is outside all segments."""
    for c in chords:
        if c.start <= t < c.end:
            return c
    return None


def _snap(pitch: int, chord: ChordSegment) -> int:
    """Nearest MIDI pitch whose pitch class is a chord tone of `chord`.

    Searches outward by semitone; on a tie the lower pitch is returned (it is
    checked first). Falls back to `pitch` unchanged if nothing is in range.
    """
    pcs = _tone_pcs(chord)
    for d in range(13):
        for cand in (pitch - d, pitch + d):
            if 0 <= cand <= 127 and cand % 12 in pcs:
                return cand
    return pitch
