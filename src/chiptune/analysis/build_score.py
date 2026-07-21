"""Assemble one Score from an audio file: separate -> transcribe -> declash -> tempo.

Role mapping: vocals -> LEAD, other -> HARMONY, bass -> BASS, drums -> PERCUSSION.
When `include_vocals` is False there is no vocal stem to carry the melody, so the
LEAD is derived from the top (skyline) voice of the `other` transcription.

Two behaviours that are easy to get subtly wrong, both locked by tests:
  * Tempo is estimated from the ORIGINAL MIX (loaded mono here), never from a
    single stem's less-reliable tempo track.
  * Harmony declash (config-gated `harmony_declash`): a HARMONY note overlapping a
    LEAD note within `declash_semitones` is pushed down one octave, so two square
    waves don't sit at near-unison and muddy the mix. A pure NoteEvent-list
    transform - the proven allocator is left untouched.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from chiptune.analysis.drums import transcribe_drums
from chiptune.analysis.separate import STEM_SR, separate_stems
from chiptune.analysis.tempo import estimate_grid
from chiptune.analysis.transcribe import transcribe_pitched
from chiptune.analysis.vocals import transcribe_vocals
from chiptune.config import Config
from chiptune.score import NoteEvent, Role, Score


def _load_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load the original mix as mono float32 at its native sample rate."""
    y, sr = sf.read(audio_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y), int(sr)


def _overlaps(a: NoteEvent, b: NoteEvent) -> bool:
    return a.start < b.end and b.start < a.end


def _skyline_lead(notes: list[NoteEvent]) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Split `other` notes into (LEAD, HARMONY): a note is LEAD iff no
    concurrently-sounding note has a strictly higher pitch (the top voice).

    The documented include_vocals=False fallback - the melody usually rides the
    top of `other` when there is no vocal stem to lead with.
    """
    lead, harmony = [], []
    for n in notes:
        covered = any(m is not n and m.pitch > n.pitch and _overlaps(n, m) for m in notes)
        if covered:
            harmony.append(n)
        else:
            lead.append(replace(n, role=Role.LEAD))
    return lead, harmony


def declash_harmony(notes: list[NoteEvent], declash_semitones: int) -> list[NoteEvent]:
    """Push each HARMONY note that overlaps a LEAD note within `declash_semitones`
    down one octave (pitch - 12). Pure transform: leads and non-clashing notes
    pass through unchanged. A push that would underflow MIDI 0 is skipped.
    """
    leads = [n for n in notes if n.role is Role.LEAD]
    out: list[NoteEvent] = []
    for n in notes:
        clashes = n.role is Role.HARMONY and any(
            _overlaps(n, lead) and abs(n.pitch - lead.pitch) <= declash_semitones
            for lead in leads
        )
        if clashes and n.pitch - 12 >= 0:
            n = replace(n, pitch=n.pitch - 12)
        out.append(n)
    return out


def build_score(audio_path, cfg: Config, cache_dir=None) -> Score:
    """Separate `audio_path`, transcribe each stem to its role, apply config-gated
    harmony declash, estimate tempo from the original mix, and assemble one Score.
    """
    audio_path = Path(audio_path)
    a = cfg.analysis

    stems = separate_stems(audio_path, cache_dir=cache_dir)
    sr = STEM_SR

    bass = transcribe_pitched(stems["bass"], sr, Role.BASS, min_duration=a.min_note_seconds)
    other = transcribe_pitched(stems["other"], sr, Role.HARMONY, min_duration=a.min_note_seconds)
    drums = transcribe_drums(
        stems["drums"], sr,
        a.kick_band_hz, a.hat_band_hz, a.kick_low_frac_min, a.hat_high_frac_min,
        backtrack=a.onset_backtrack,
    )

    if a.include_vocals:
        lead = transcribe_vocals(
            stems["vocals"], sr, a.vocal_fmin, a.vocal_fmax, min_duration=a.min_note_seconds
        )
        harmony = other
    else:
        lead, harmony = _skyline_lead(other)

    notes = lead + harmony + bass + drums
    declash_pushed = 0
    if a.harmony_declash:
        declashed = declash_harmony(notes, a.declash_semitones)
        # Count how many harmony notes were actually pushed down: declash is
        # "the single most likely thing to need tuning at the listen", so make
        # its effect an observable number rather than a black box.
        declash_pushed = sum(1 for b, c in zip(notes, declashed) if b.pitch != c.pitch)
        notes = declashed

    # Tempo from the ORIGINAL MIX, not any single stem.
    mono, orig_sr = _load_mono(audio_path)
    grid = estimate_grid(mono, orig_sr)

    duration = max((n.end for n in notes), default=0.0)
    score = Score(tempo=grid, notes=notes, duration=duration)

    counts = {r: sum(1 for n in notes if n.role is r) for r in Role}
    print(
        f"chiptune.analysis.build_score: {grid.bpm:.1f} BPM; "
        f"lead={counts[Role.LEAD]} harmony={counts[Role.HARMONY]} "
        f"bass={counts[Role.BASS]} percussion={counts[Role.PERCUSSION]} "
        f"declash_pushed={declash_pushed}",
        file=sys.stderr,
    )
    return score
