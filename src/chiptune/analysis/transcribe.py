"""Pitched-instrument transcription (bass, other) via basic-pitch.

basic-pitch's `predict` only accepts a file path, not an in-memory array, so
the stem is written to a temp wav first. `_run_basic_pitch` is the single seam
that touches the model - everything else is pure tuple -> `NoteEvent` mapping,
which keeps the mapping unit-testable without paying for a model load.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from chiptune.score import NoteEvent, Role


def _run_basic_pitch(wav_path: str) -> list[tuple]:
    from basic_pitch.inference import predict

    _model_output, _midi_data, note_events = predict(wav_path)
    return note_events


def transcribe_pitched(
    stem: np.ndarray, sr: int, role: Role, min_duration: float = 0.0
) -> list[NoteEvent]:
    """Transcribe a monophonic-or-polyphonic pitched stem to `NoteEvent`s via basic-pitch.

    Notes shorter than `min_duration` seconds are dropped.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "stem.wav"
        sf.write(wav_path, stem, sr)
        raw_notes = _run_basic_pitch(str(wav_path))

    notes = []
    for start, end, pitch, amplitude, _pitch_bends in raw_notes:
        if end - start < min_duration:
            continue
        velocity = int(np.clip(amplitude * 127, 1, 127))
        notes.append(
            NoteEvent(pitch=int(pitch), start=float(start), end=float(end), velocity=velocity, role=role)
        )
    return notes
