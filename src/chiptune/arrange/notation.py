"""Parse the LLM arranger's key-relative notation into NoteEvents.

The model writes scale degrees against a key it declares (KEY: line), so it can
never emit an out-of-key pitch and octave/register are our job. Malformed tokens
are dropped (with a stderr warning); only genuinely empty/keyless output raises
NotationError so the caller can fall back to the heuristic arranger.
"""
from __future__ import annotations

import math
import sys

from ..score import NoteEvent, Percussion, Role, TempoGrid

_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]
_ROOTS = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,
          "G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11}
_VOICE_ROLE = {"LEAD": Role.LEAD, "HARM": Role.HARMONY, "BASS": Role.BASS, "DRUMS": Role.PERCUSSION}
_DRUMS = {"K": Percussion.KICK, "S": Percussion.SNARE, "H": Percussion.HAT}


class NotationError(Exception):
    """The arranger output had no usable content; caller should fall back."""


def _parse_key(line: str) -> tuple[int, bool]:
    parts = line.split(":", 1)[1].split()
    root = _ROOTS[parts[0].strip().upper()]
    is_minor = parts[1].strip().lower().startswith("min")
    return root, is_minor


def _degree_to_semitone(token: str, is_minor: bool) -> int:
    acc = 0
    while token and token[0] in "b#":
        acc += -1 if token[0] == "b" else 1
        token = token[1:]
    octs = token.count("'") - token.count(",")
    core = token.replace("'", "").replace(",", "")
    degree = int(core)  # raises ValueError on garbage -> caller skips
    if not 1 <= degree <= 7:
        raise ValueError(degree)
    scale = _MINOR if is_minor else _MAJOR
    return scale[degree - 1] + acc + 12 * octs


def parse_arrangement(text: str, grid: TempoGrid, octaves: dict[str, int]) -> list[NoteEvent]:
    spb = grid.seconds_per_beat
    key = None
    voices: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = line.split(":", 1)[0].strip().upper()
        if head == "KEY":
            try:
                key = _parse_key(line)
            except (KeyError, IndexError):
                key = None
        elif head in _VOICE_ROLE:
            body = line.split(":", 1)[1] if ":" in line else ""
            # A voice may span multiple lines (an LLM wrapping a long part);
            # concatenate rather than let a later line clobber an earlier one.
            voices[head] = f"{voices[head]} {body}" if head in voices else body
    if key is None or not voices:
        raise NotationError("no KEY or no voice lines")

    root, is_minor = key
    notes: list[NoteEvent] = []
    dropped = 0
    for voice, body in voices.items():
        role = _VOICE_ROLE[voice]
        base = 12 * (octaves.get(voice, 4) + 1) + root
        t = 0.0
        for tok in body.replace("|", " ").split():
            if ":" not in tok:
                dropped += 1
                continue
            sym, _, durs = tok.partition(":")
            try:
                dur = float(durs)
                if not math.isfinite(dur) or dur <= 0:
                    raise ValueError
            except ValueError:
                dropped += 1
                continue
            end = t + dur * spb
            if role is Role.PERCUSSION:
                if sym.upper() != "R":
                    for ch in sym.upper():
                        kind = _DRUMS.get(ch)
                        if kind is None:
                            dropped += 1
                            continue
                        notes.append(NoteEvent(pitch=38, start=t, end=end, velocity=100,
                                               role=Role.PERCUSSION, percussion=kind))
            elif sym.upper() != "R":
                try:
                    pitch = base + _degree_to_semitone(sym, is_minor)
                except (ValueError, IndexError):
                    dropped += 1
                    t = end
                    continue
                while pitch > 127:
                    pitch -= 12
                while pitch < 0:
                    pitch += 12
                notes.append(NoteEvent(pitch=pitch, start=t, end=end, velocity=100, role=role))
            t = end

    if not notes:
        raise NotationError("no notes parsed from any voice")
    if dropped:
        print(f"chiptune.ai_arranger: dropped {dropped} malformed token(s)", file=sys.stderr)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
