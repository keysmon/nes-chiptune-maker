"""LLM-as-arranger: prompt a model with the song's melody + tempo, parse its
scale-degree arrangement into a Score. Any failure falls back to the heuristic
arranger, so AI mode never crashes or produces silence.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

from ..config import AIConfig
from ..score import NoteEvent, Role, Score, TempoGrid
from .notation import NotationError, parse_arrangement

_DEFAULT_OCTAVES = {"LEAD": 4, "HARM": 3, "BASS": 2}
_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

_SYSTEM = (
    "You are an expert NES (2A03) chiptune arranger. The chip has exactly 4 voices: "
    "Pulse1 (LEAD), Pulse2 (HARM), Triangle (BASS), Noise (DRUMS: K kick, S snare, H hat). "
    "Given a song's melody and tempo, write a tight, musical 4-voice arrangement. "
    "PRESERVE the given melody as the LEAD (you may simplify, do not replace the tune). "
    "Write every pitched voice as scale DEGREES 1-7 (b/# accidentals, ' up-octave , down-octave) "
    "relative to a key you choose and declare. "
    "Scale degrees are relative to the KEY'S MODE: in a minor key, 1-7 are the natural minor "
    "scale (so 3 is the minor third, 6 the minor sixth, 7 the minor seventh) - do NOT write "
    "b3/b6/b7 for those. Use b/# ONLY for notes chromatic to the declared key. "
    "Keep it sparse and deliberate - space is good. "
    "Output ONLY these lines, nothing else:\n"
    "KEY: <root> <maj|min>\nLEAD: <deg:beats> ...\nHARM: ...\nBASS: ...\nDRUMS: <KSH:beats> ...\n"
    "Durations are in beats. Use R for rests."
)


def format_prompt(score: Score) -> str:
    lead = sorted((n for n in score.notes if n.role is Role.LEAD), key=lambda n: n.start)
    spb = score.tempo.seconds_per_beat
    mel = " ".join(f"{_NAMES[n.pitch % 12]}{n.pitch // 12 - 1}:{round((n.end - n.start) / spb, 2)}"
                   for n in lead) or "(none)"
    return (f"TEMPO: {round(score.tempo.bpm, 1)} BPM\n"
            f"MELODY (note:beats): {mel}\n\n"
            "Arrange this for the NES 4 voices now.")


def _call_llm(prompt: str, cfg: AIConfig) -> str:
    from openai import OpenAI  # lazy: only needed in AI mode
    key = os.environ.get(cfg.api_key_env, "")
    client = OpenAI(base_url=cfg.base_url, api_key=key or "none")
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def arrange(score: Score, ai_cfg: AIConfig, octaves: dict[str, int] | None,
            heuristic_fn: Callable[[], Score]) -> Score:
    octaves = octaves or _DEFAULT_OCTAVES
    try:
        text = _call_llm(format_prompt(score), ai_cfg)
        notes = parse_arrangement(text, score.tempo, octaves)
    except NotationError as exc:
        print(f"chiptune.ai_arranger: unparseable output ({exc}); using heuristic", file=sys.stderr)
        return heuristic_fn()
    except Exception as exc:  # network / SDK / auth - degrade, don't crash
        print(f"chiptune.ai_arranger: LLM call failed ({exc}); using heuristic", file=sys.stderr)
        return heuristic_fn()
    duration = max((n.end for n in notes), default=score.duration)
    return Score(tempo=score.tempo, notes=notes, duration=duration)
