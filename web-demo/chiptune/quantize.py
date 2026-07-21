"""Snap note onsets to the tempo grid and pitches to semitones.

Mandatory, not optional: raw pitch tracking produces vibrato, glissando, and
human timing drift. A pulse channel has no vibrato of its own, so un-quantized
pitch sounds unstable, and un-quantized onsets fight the 60 Hz frame clock.
Strength is a knob because over-quantizing removes groove.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .score import NoteEvent, Score


def _snap(t: float, step: float, offset: float, strength: float) -> float:
    if step <= 0:
        raise ValueError(f"step must be positive (got {step})")
    snapped = offset + round((t - offset) / step) * step
    return t + strength * (snapped - t)


def quantize_score(
    score: Score,
    subdivision: int,
    strength: float,
    min_duration: float,
) -> Score:
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1] (got {strength})")
    if subdivision <= 0:
        raise ValueError(f"subdivision must be positive (got {subdivision})")
    if min_duration <= 0:
        raise ValueError(f"min_duration must be positive (got {min_duration})")

    step = score.tempo.seconds_per_beat * 4.0 / subdivision
    offset = score.tempo.offset

    out: list[NoteEvent] = []
    for n in score.notes:
        start = _snap(n.start, step, offset, strength)
        end = _snap(n.end, step, offset, strength)
        if end - start < min_duration:
            end = start + min_duration
            # `start + min_duration` can round down by one ULP (e.g. 0.125 + 0.02
            # == 0.145, but 0.145 - 0.125 == 0.019999999999999997 < 0.02), which
            # would violate the min-duration floor this branch exists to enforce.
            # Bump to the next representable float until the floor actually holds.
            while end - start < min_duration:
                end = math.nextafter(end, math.inf)
        out.append(replace(n, pitch=int(round(n.pitch)), start=start, end=end))

    out.sort(key=lambda n: (n.start, n.pitch))
    return Score(tempo=score.tempo, notes=out, duration=score.duration)
