"""Note-selection harmony comp: keep the important real harmony notes, pruned
to a sparsity budget and snapped to the detected chords, instead of a fixed
arpeggio. Design: docs/superpowers/specs/2026-07-22-note-selection-comp-design.md
"""
from __future__ import annotations

from ..score import NoteEvent, Role, TempoGrid
from ..analysis.chords import ChordSegment

_HARMONY_VELOCITY = 80
# Taste-tunable ranking weights for `_importance`. Kept in code (not
# config/nes.toml) per the plan's v1 YAGNI decision - these shape relative
# ranking, not an audible parameter a listener tunes directly.
_IMPORTANCE_BASE = 0.5
_OFFCHORD_PENALTY = 0.4


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
    fit = 1.0 if (chord is not None and note.pitch % 12 in _tone_pcs(chord)) else _OFFCHORD_PENALTY
    return dur * (_IMPORTANCE_BASE + loud) * fit


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


def _enforce_mono(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Sort by start and clamp each note's end to the next note's start, dropping
    any note that collapses to zero length. Guarantees strict monophony."""
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


def select_comp(
    harmony_notes: list[NoteEvent],
    chords: list[ChordSegment],
    lead_notes: list[NoteEvent],
    grid: TempoGrid,
    cfg,
) -> list[NoteEvent]:
    """Select and clean a monophonic harmony comp from candidate `harmony_notes`.

    Greedy by importance, spaced at least `cfg.select_min_gap` apart on start
    times; optionally rest under an active LEAD note; snap each kept note to the
    nearest detected chord tone; voice-lead octaves for smooth motion; enforce
    monophony. Any chord segment left with no emitted note gets a voice-led
    chord-tone fallback so the comp never goes silent on a chord.
    """
    if not chords:
        return []
    ranked = sorted(
        harmony_notes,
        key=lambda n: _importance(n, _chord_at(chords, n.start)),
        reverse=True,
    )
    kept: list[NoteEvent] = []
    for n in ranked:
        if all(abs(n.start - m.start) >= cfg.select_min_gap for m in kept):
            kept.append(n)
    if cfg.harmony_rest_on_busy_melody:
        kept = [n for n in kept if not _lead_active(lead_notes, n.start, n.end)]
    kept.sort(key=lambda n: n.start)

    emitted: list[NoteEvent] = []
    prev_pitch: int | None = None
    for n in kept:
        chord = _chord_at(chords, n.start)
        if chord is None:
            continue
        pitch = _voice_lead(_snap(n.pitch, chord), prev_pitch)
        emitted.append(NoteEvent(pitch=pitch, start=n.start, end=n.end,
                                 velocity=_HARMONY_VELOCITY, role=Role.HARMONY))
        prev_pitch = pitch

    # Fallback: any chord segment with no emitted note gets one voice-led chord
    # tone (root snapped into the chord, at the configured octave) so the comp
    # never goes silent on a chord - the safety net for poor transcriptions.
    # Resting under a busy melody still wins over the fallback: giving the
    # tune space is a deliberate choice, not an untrustworthy transcription,
    # so a chord with an active LEAD is not force-filled either (this is the
    # brief-code fix noted in batchB-report.md - see "Brief defect found").
    def _covered(c: ChordSegment) -> bool:
        return any(c.start <= e.start < c.end for e in emitted)

    spb = grid.seconds_per_beat
    for c in chords:
        if _covered(c):
            continue
        end = min(c.start + spb, c.end)
        if end <= c.start:
            continue
        if cfg.harmony_rest_on_busy_melody and _lead_active(lead_notes, c.start, end):
            continue
        root_midi = (cfg.chord_octave + 1) * 12 + c.root
        pitch = _voice_lead(_snap(root_midi, c), prev_pitch)
        emitted.append(NoteEvent(pitch=pitch, start=c.start, end=end,
                                 velocity=_HARMONY_VELOCITY, role=Role.HARMONY))
        prev_pitch = pitch
    return _enforce_mono(emitted)
