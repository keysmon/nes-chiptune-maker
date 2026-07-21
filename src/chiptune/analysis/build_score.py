"""Assemble one Score from an audio file: separate -> transcribe -> arrange -> tempo.

Role mapping: vocals -> LEAD, bass -> BASS, drums -> PERCUSSION. HARMONY depends
on `[arrange].harmony_mode`:
  * "chords" (default) - the sparse arranger. HARMONY comes from
    `chiptune.analysis.chords.detect_chords` on the ORIGINAL MIX, comped by
    `chiptune.arrange.chord_comp.comp_chords`. The "other" stem is never
    basic-pitch transcribed for HARMONY in this mode - that transcription is
    exactly the "hundreds of near-simultaneous noisy notes" mud problem this
    mode replaces.
  * "transcribe" - the original path: basic-pitch on the "other" stem,
    reproduced exactly (byte-for-byte call arguments) for backward compat.
When `include_vocals` is False there is no vocal stem to carry the melody, so
the LEAD is derived from the top (skyline) voice of an `other` transcription -
this is still needed in "chords" mode too (there is no other melody source),
but its covered/lower half is discarded rather than merged into HARMONY: the
"other" stem is still transcribed in that one combination, but its output
never populates HARMONY, which is the actual mud this mode fixes.

LEAD and BASS are always passed through the sparse arranger's thinning
transforms (`chiptune.arrange.sparse`), in both harmony_mode settings.

Behaviours easy to get subtly wrong, all locked by tests:
  * Tempo (and, in "chords" mode, the chord-detection beat grid) is estimated
    from the ORIGINAL MIX (loaded mono here), never from a single stem's
    less-reliable tempo track.
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

from chiptune.analysis.chords import detect_chords
from chiptune.analysis.density import score_density
from chiptune.analysis.drums import transcribe_drums
from chiptune.analysis.separate import STEM_SR, separate_stems
from chiptune.analysis.tempo import estimate_grid
from chiptune.analysis.transcribe import transcribe_pitched
from chiptune.analysis.vocals import transcribe_vocals
from chiptune.arrange.chord_comp import comp_chords
from chiptune.arrange.sparse import rest_harmony_on_busy_melody, simplify_bass, thin_melody
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
    """Separate `audio_path`, transcribe/arrange each role, apply config-gated
    harmony declash, estimate tempo from the original mix, and assemble one Score.
    """
    audio_path = Path(audio_path)
    a = cfg.analysis
    arr = cfg.arrange

    stems = separate_stems(audio_path, cache_dir=cache_dir)
    sr = STEM_SR

    bass = transcribe_pitched(stems["bass"], sr, Role.BASS, min_duration=a.min_note_seconds)
    drums = transcribe_drums(
        stems["drums"], sr,
        a.kick_band_hz, a.hat_band_hz, a.kick_low_frac_min, a.hat_high_frac_min,
        backtrack=a.onset_backtrack,
    )

    # Tempo (and, in "chords" mode, the chord-detection beat grid) both come
    # from the ORIGINAL MIX, not any single stem.
    mono, orig_sr = _load_mono(audio_path)
    grid = estimate_grid(mono, orig_sr)

    if a.include_vocals:
        lead = transcribe_vocals(
            stems["vocals"], sr, a.vocal_fmin, a.vocal_fmax, min_duration=a.min_note_seconds
        )
        skyline_harmony = None
    else:
        # The only melody source when there's no vocal stem, in either
        # harmony_mode - the "other" transcription's covered/lower half
        # (skyline_harmony) is only actually used as HARMONY in "transcribe"
        # mode; see below.
        other = transcribe_pitched(stems["other"], sr, Role.HARMONY, min_duration=a.min_note_seconds)
        lead, skyline_harmony = _skyline_lead(other)

    if arr.harmony_mode == "chords":
        chords = detect_chords(mono, orig_sr, grid, smooth_beats=arr.chord_smooth_beats)
        harmony = comp_chords(
            chords,
            pattern=arr.chord_comp_pattern,
            subdivision=arr.chord_subdivision,
            octave=arr.chord_octave,
            tones=arr.chord_tones,
            grid=grid,
        )
    elif arr.harmony_mode == "transcribe":
        if a.include_vocals:
            harmony = transcribe_pitched(
                stems["other"], sr, Role.HARMONY, min_duration=a.min_note_seconds
            )
        else:
            harmony = skyline_harmony
    else:
        raise ValueError(f"unknown [arrange].harmony_mode: {arr.harmony_mode!r}")

    lead = thin_melody(lead, arr.melody_min_seconds)
    bass = simplify_bass(bass, arr.bass_min_seconds)
    if arr.harmony_rest_on_busy_melody:
        harmony = rest_harmony_on_busy_melody(harmony, lead)

    notes = lead + harmony + bass + drums
    declash_pushed = 0
    if a.harmony_declash:
        declashed = declash_harmony(notes, a.declash_semitones)
        # Count how many harmony notes were actually pushed down: declash is
        # "the single most likely thing to need tuning at the listen", so make
        # its effect an observable number rather than a black box.
        declash_pushed = sum(1 for b, c in zip(notes, declashed) if b.pitch != c.pitch)
        notes = declashed

    duration = max((n.end for n in notes), default=0.0)
    heuristic_score = Score(tempo=grid, notes=notes, duration=duration)

    counts = {r: sum(1 for n in notes if n.role is r) for r in Role}
    print(
        f"chiptune.analysis.build_score: {grid.bpm:.1f} BPM; harmony_mode={arr.harmony_mode}; "
        f"lead={counts[Role.LEAD]} harmony={counts[Role.HARMONY]} "
        f"bass={counts[Role.BASS]} percussion={counts[Role.PERCUSSION]} "
        f"declash_pushed={declash_pushed}",
        file=sys.stderr,
    )
    density = score_density(heuristic_score, frame_rate=cfg.frame_rate)
    print(
        f"chiptune.analysis.build_score: density mean_simultaneous="
        f"{density['mean_simultaneous']:.2f} lead_active={density['per_role_active'][Role.LEAD]:.2f} "
        f"harmony_active={density['per_role_active'][Role.HARMONY]:.2f} "
        f"bass_active={density['per_role_active'][Role.BASS]:.2f}",
        file=sys.stderr,
    )

    # "ai" = hand the heuristic Score (melody/tempo source AND fallback) to the
    # LLM arranger; "heuristic" (default) leaves existing behavior unchanged.
    if arr.arrange_mode == "ai":
        from chiptune.arrange.ai_arranger import arrange as ai_arrange

        return ai_arrange(heuristic_score, cfg.ai, None, lambda: heuristic_score)
    return heuristic_score
