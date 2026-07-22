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


def _importance(note: NoteEvent, chord: ChordSegment | None) -> float:
    """Rank a candidate harmony note: longer, louder, and chord-fitting notes
    score higher. Off-chord notes are penalized but not zeroed (they can still
    win a slot if nothing better exists, and get snapped afterward)."""
    dur = note.end - note.start
    loud = note.velocity / 127.0
    fit = 1.0 if (chord is not None and note.pitch % 12 in _tone_pcs(chord)) else 0.4
    return dur * (0.5 + loud) * fit


def _lead_active(lead: list[NoteEvent], t0: float, t1: float) -> bool:
    """True if any LEAD note overlaps the half-open interval [t0, t1)."""
    return any(n.start < t1 and n.end > t0 for n in lead)


def _voice_lead(pitch: int, prev: int | None) -> int:
    """Shift `pitch` by whole octaves to sit as close as possible to `prev`,
    staying in MIDI range. Returns `pitch` unchanged when there is no previous."""
    if prev is None:
        return pitch
    best = pitch
    while best - 12 >= 0 and abs((best - 12) - prev) < abs(best - prev):
        best -= 12
    while best + 12 <= 127 and abs((best + 12) - prev) < abs(best - prev):
        best += 12
    return best
