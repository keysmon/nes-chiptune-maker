"""Chord -> arpeggio expansion.

A single 2A03 channel cannot play a chord. NES composers cycled one channel
through the chord tones every frame or two, fast enough that the ear fuses them
into harmony. The rate matters: too slow and you hear a cycling melody, too fast
and it buzzes. That is why arpeggio_frames is a config knob.
"""
from __future__ import annotations

from ..score import NoteEvent


def _active_pitches(notes: list[NoteEvent], t: float) -> list[int]:
    return sorted(n.pitch for n in notes if n.start <= t < n.end)


def arpeggiate(
    notes: list[NoteEvent],
    n_frames: int,
    frame_rate: float,
    arpeggio_frames: int,
) -> list[int | None]:
    if arpeggio_frames < 1:
        raise ValueError(f"arpeggio_frames must be >= 1, got {arpeggio_frames}")

    out: list[int | None] = []
    step = 0            # index into the current chord
    dwell = 0           # frames spent on the current tone
    prev_chord: tuple[int, ...] = ()

    for f in range(n_frames):
        t = f / frame_rate
        chord = tuple(_active_pitches(notes, t))

        if not chord:
            out.append(None)
            step, dwell, prev_chord = 0, 0, ()
            continue

        if chord != prev_chord:
            # New chord: restart the cycle so the root is heard first.
            step, dwell = 0, 0
            prev_chord = chord

        out.append(chord[step % len(chord)])

        dwell += 1
        if dwell >= arpeggio_frames:
            dwell = 0
            step += 1

    return out
