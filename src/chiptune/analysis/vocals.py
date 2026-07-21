"""Vocal monophonic transcription via `librosa.pyin`.

Vocals are monophonic (one pitch at a time), unlike bass/other which basic-pitch
handles polyphonically, so a lighter pitch tracker suffices here and avoids a
second model load. `pyin` returns one f0/voiced/probability triple per frame;
this module quantizes each voiced frame to the nearest semitone and merges
consecutive same-pitch voiced frames into `NoteEvent`s.
"""
from __future__ import annotations

import numpy as np
import librosa

from chiptune.score import NoteEvent, Role

HOP_LENGTH = 512


def transcribe_vocals(
    stem: np.ndarray, sr: int, fmin: float, fmax: float, min_duration: float = 0.06
) -> list[NoteEvent]:
    """Transcribe a monophonic vocal stem to `NoteEvent`s (role=LEAD).

    Returns `[]` if no frame is voiced - not an error, just a quiet/empty
    vocal stem. Notes shorter than `min_duration` seconds are dropped.
    """
    f0, voiced, vprob = librosa.pyin(stem, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP_LENGTH)
    # A frame flagged voiced=True but with a non-finite f0 (NaN/inf) would cast
    # to pitch 0 below and emit a bogus MIDI-0 LEAD note; treat it as unvoiced
    # instead. pyin already sets f0=NaN whenever voiced=False, so this mask
    # equals `voiced` on current librosa - it's a guard against a
    # future/edge-case mismatch, not a behavior change today.
    voiced = voiced & np.isfinite(f0)
    if not np.any(voiced):
        return []

    pitches = np.full(len(f0), -1, dtype=np.int64)
    pitches[voiced] = np.round(librosa.hz_to_midi(f0[voiced])).astype(np.int64)

    notes = []
    run_start = None
    run_pitch = None

    def close_run(end_idx: int) -> None:
        # run_start and run_pitch are always set (or cleared) together below,
        # but pyright can't infer that correlation across two separate
        # closured variables - check both explicitly so it narrows run_pitch
        # from `int | None` to `int` for the np.clip call.
        if run_start is None or run_pitch is None:
            return
        start = run_start * HOP_LENGTH / sr
        end = end_idx * HOP_LENGTH / sr
        if end - start < min_duration:
            return
        velocity = int(np.clip(vprob[run_start:end_idx].mean() * 127, 1, 127))
        notes.append(
            NoteEvent(
                pitch=int(np.clip(run_pitch, 0, 127)),
                start=float(start),
                end=float(end),
                velocity=velocity,
                role=Role.LEAD,
            )
        )

    for i, is_voiced in enumerate(voiced):
        pitch = int(pitches[i]) if is_voiced else None
        if pitch != run_pitch:
            # Voicing resumed, stopped, or pitch changed: close the current
            # run (if any) and open a new one (if this frame is voiced).
            close_run(i)
            run_start = i if is_voiced else None
            run_pitch = pitch
    close_run(len(voiced))

    return notes
