"""Chord-progression -> clean arpeggiated harmony.

Companion to `chiptune.analysis.chords`: instead of basic-pitch transcribing
the "other" stem note-by-note (hundreds of overlapping, noisy notes), comp
each detected chord as a short, evenly-spaced arpeggio of its own chord
tones. One 2A03 channel is still monophonic, so the notes this emits never
overlap - the arpeggiate step in `chiptune.arrange.arpeggio` handles the
"more notes than the channel can hold at once" case; this module only ever
plays one chord tone at a time by construction.

Octave numbering follows scientific pitch notation (middle C = C4 = MIDI 60),
i.e. MIDI base for a pitch class in `octave` is `(octave + 1) * 12`.
"""
from __future__ import annotations

from ..score import NoteEvent, Role, TempoGrid
from ..analysis.chords import ChordSegment

_HARMONY_VELOCITY = 80
# A comped note this short or shorter is dropped rather than emitted: when a
# segment's duration isn't an exact multiple of `step`, the last note in the
# segment is a near-zero-length remainder, not a musical note - and NoteEvent
# happily accepts it (end > start is all it requires), so it would otherwise
# reach the pipeline as a degenerate blip.
_MIN_COMP_NOTE_SECONDS = 0.01


def _fold_pitch(pitch: int) -> int:
    """Fold a pitch into the valid MIDI range [0, 127] by whole octaves, so an
    extreme `chord_octave` (config-exposed, e.g. via the web UI) can never push a
    chord tone past 127 and crash NoteEvent. Sane octaves are a no-op."""
    while pitch > 127:
        pitch -= 12
    while pitch < 0:
        pitch += 12
    return pitch


def _chord_tones(base_pitch: int, third_interval: int, tones: int) -> list[int]:
    """`tones` chord-tone MIDI pitches: root, third, fifth, then repeat an octave up.

    `base_pitch` is the root's MIDI pitch (octave already applied).
    """
    if tones < 1:
        raise ValueError(f"tones must be >= 1 (got {tones})")
    intervals = (0, third_interval, 7)
    out = []
    for i in range(tones):
        octave_shift, idx = divmod(i, len(intervals))
        out.append(base_pitch + intervals[idx] + 12 * octave_shift)
    return out


def _pattern_sequence(
    pattern: str, base_pitch: int, third_interval: int, tones: int
) -> list[int]:
    if pattern == "root_fifth":
        # Ignores `tones` by design (brief: alternates root/fifth regardless
        # of how many chord tones are otherwise in play).
        return [base_pitch, base_pitch + 7]

    chord_tones = _chord_tones(base_pitch, third_interval, tones)
    if pattern == "up":
        return chord_tones
    if pattern == "updown":
        if len(chord_tones) <= 2:
            return chord_tones
        # Ascend then descend without repeating either endpoint back-to-back.
        return chord_tones + chord_tones[-2:0:-1]
    raise ValueError(f"unknown comp pattern: {pattern!r}")


def comp_chords(
    chords: list[ChordSegment],
    pattern: str,
    subdivision: int,
    octave: int,
    tones: int,
    grid: TempoGrid,
) -> list[NoteEvent]:
    """Comp each chord segment as a short arpeggio of its chord tones.

    `subdivision` is notes per beat (not a note-value denominator like
    `TempoGrid.subdivision_times` uses) - subdivision=3 over a 1-beat
    segment emits exactly 3 evenly-spaced notes.
    """
    if subdivision < 1:
        raise ValueError(f"subdivision must be >= 1 (got {subdivision})")
    step = grid.seconds_per_beat / subdivision

    notes: list[NoteEvent] = []
    for seg in chords:
        base_pitch = (octave + 1) * 12 + seg.root
        third_interval = 3 if seg.is_minor else 4
        sequence = [_fold_pitch(p) for p in
                    _pattern_sequence(pattern, base_pitch, third_interval, tones)]
        if not sequence:
            continue

        t = seg.start
        i = 0
        while t < seg.end:
            note_end = min(t + step, seg.end)
            if note_end - t >= _MIN_COMP_NOTE_SECONDS:
                notes.append(
                    NoteEvent(
                        pitch=sequence[i % len(sequence)],
                        start=t,
                        end=note_end,
                        velocity=_HARMONY_VELOCITY,
                        role=Role.HARMONY,
                    )
                )
            t = note_end
            i += 1

    return notes
