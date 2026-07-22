"""LLM-as-arranger: prompt a model with the song's melody + tempo, parse its
scale-degree arrangement into a Score. Any failure falls back to the heuristic
arranger, so AI mode never crashes or produces silence.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

from ..config import AIConfig
from ..score import Role, Score
from .notation import NotationError, parse_arrangement

_DEFAULT_OCTAVES = {"LEAD": 4, "HARM": 3, "BASS": 2}
_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Arrangement fullness dial: the same melody+chords can be arranged sparse (good for
# a busy/muddy source) or full (good for a thin one). Config-selected per song.
_DENSITY_HINT = {
    "sparse":   "DENSITY: SPARSE - lots of space, minimal HARM and BASS, let the melody breathe.",
    "balanced": "DENSITY: BALANCED - support the melody without cluttering it.",
    "full":     "DENSITY: FULL - rich, active HARM and BASS throughout; keep the accompaniment present.",
}

_SYSTEM = (
    "You are an expert NES (2A03) chiptune arranger. The chip has exactly 4 voices: "
    "Pulse1 (LEAD), Pulse2 (HARM), Triangle (BASS), Noise (DRUMS: K kick, S snare, H hat). "
    "Given a song's melody, chord progression, bass line, and tempo, write a tight, musical "
    "4-voice arrangement. "
    "PRESERVE the given melody as the LEAD (you may simplify, do not replace the tune). "
    "Base HARM and the BASS roots on the given CHORDS - that is the song's real harmony; "
    "do not invent a different progression. "
    "Write every pitched voice as scale DEGREES 1-7 (b/# accidentals, ' up-octave , down-octave) "
    "relative to a key you choose and declare. "
    "Scale degrees are relative to the KEY'S MODE: in a minor key, 1-7 are the natural minor "
    "scale (so 3 is the minor third, 6 the minor sixth, 7 the minor seventh) - do NOT write "
    "b3/b6/b7 for those. Use b/# ONLY for notes chromatic to the declared key. "
    "Keep it sparse and deliberate - space is good. "
    "Output ONLY these lines, nothing else:\n"
    "KEY: <root> <maj|min>\nLEAD: <deg:beats> ...\nHARM: ...\nBASS: ...\nDRUMS: <KSH:beats> ...\n"
    "Durations are in beats. Use R for rests. "
    "Each voice's durations MUST sum to about the total beats stated for the song - "
    "match the song's length and NEVER exceed it."
)


def format_prompt(score: Score, chords: list | None = None, density: str = "balanced") -> str:
    """Summarize the analyzed song for the LLM: melody, the DETECTED chord
    progression, the real bass line, and drum density - so it arranges the actual
    song instead of guessing the harmony from the melody alone. `density` selects
    how full the arrangement should be."""
    spb = score.tempo.seconds_per_beat

    def fmt(notes):
        return " ".join(f"{_NAMES[n.pitch % 12]}{n.pitch // 12 - 1}:{round((n.end - n.start) / spb, 2)}"
                        for n in notes) or "(none)"

    lead = sorted((n for n in score.notes if n.role is Role.LEAD), key=lambda n: n.start)
    bass = sorted((n for n in score.notes if n.role is Role.BASS), key=lambda n: n.start)
    total_beats = max(1, round(score.duration / spb)) if spb > 0 else 1
    bars = max(1, round(total_beats / score.tempo.beats_per_bar))
    n_drums = sum(1 for n in score.notes if n.role is Role.PERCUSSION)

    lines = [f"TEMPO: {round(score.tempo.bpm, 1)} BPM",
             f"LENGTH: about {round(score.duration)} s = ~{bars} bars = ~{total_beats} beats total."]
    if chords:
        prog = " ".join(f"{_NAMES[c.root]}{'m' if c.is_minor else ''}:{max(1, round((c.end - c.start) / spb))}"
                        for c in chords)
        lines.append(f"CHORDS (name:beats): {prog}")
    lines.append(f"MELODY (note:beats): {fmt(lead)}")
    lines.append(f"BASS (note:beats): {fmt(bass)}")
    lines.append(f"DRUMS in source: {'busy' if n_drums > total_beats else 'present' if n_drums else 'none'}")
    lines.append(_DENSITY_HINT.get(density, _DENSITY_HINT["balanced"]))
    lines.append("")
    lines.append(f"Arrange this for the NES 4 voices now. Follow the CHORDS for HARM and the BASS "
                 f"roots; keep the MELODY as LEAD. Each voice's durations must sum to about "
                 f"{total_beats} beats - do NOT exceed the song's length. Match the DRUMS density above.")
    return "\n".join(lines)


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
            heuristic_fn: Callable[[], Score], chords: list | None = None) -> Score:
    octaves = octaves or _DEFAULT_OCTAVES
    # Cap the arrangement near the song length so a length-runaway (the LLM writing
    # a 590s bass line for a 30s song) is truncated to the song, not the 600s
    # buffer-safety rail. 1.5x with an 8s floor: generous enough that bar-rounding on
    # a short song is never truncated, tight enough to catch a gross (10-20x) runaway.
    # parse_arrangement further clamps this to the absolute rail.
    max_seconds = max(score.duration * 1.5, score.duration + 8.0)
    try:
        text = _call_llm(format_prompt(score, chords, ai_cfg.density), ai_cfg)
        notes = parse_arrangement(text, score.tempo, octaves, max_seconds=max_seconds)
    except NotationError as exc:
        print(f"chiptune.ai_arranger: unparseable output ({exc}); using heuristic", file=sys.stderr)
        return heuristic_fn()
    except Exception as exc:  # network / SDK / auth - degrade, don't crash
        print(f"chiptune.ai_arranger: LLM call failed ({exc}); using heuristic", file=sys.stderr)
        return heuristic_fn()
    duration = max((n.end for n in notes), default=score.duration)
    return Score(tempo=score.tempo, notes=notes, duration=duration)
