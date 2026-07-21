"""Demucs stem separation.

Splits a mixed song into four stems (drums, bass, other, vocals) via Demucs'
Python API - never the CLI, since a subprocess call can't surface a clean
Python exception and is slower to iterate on. A single separation is not
cheap (multi-second even on MPS), and the pipeline is meant to be re-run
often while tuning downstream analysis, so results are cached to disk by
content hash.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "htdemucs"
STEM_SR = 44100  # Demucs htdemucs native sample rate
STEM_NAMES = ("drums", "bass", "other", "vocals")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "stems"


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _content_hash(audio_path: Path, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(audio_path.read_bytes())
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def _load_cache(cache_file: Path) -> dict[str, np.ndarray]:
    with np.load(cache_file) as data:
        return {name: data[name] for name in STEM_NAMES}


def _to_mono_f32(wav: torch.Tensor) -> np.ndarray:
    """(channels, samples) torch tensor -> mono float32 numpy array."""
    array = wav.detach().cpu().numpy()
    mono = array.mean(axis=0) if array.ndim > 1 else array
    return np.ascontiguousarray(mono.astype(np.float32))


def separate_stems(audio_path: str | Path, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Separate `audio_path` into drums/bass/other/vocals stems.

    Each returned array is mono float32 at STEM_SR. Results are cached by
    sha256(file bytes + model name) as a `.npz` under `cache_dir` (default
    `cache/stems/`); a cache hit skips Demucs entirely.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = _content_hash(audio_path, MODEL_NAME)
    cache_file = cache_dir / f"{digest}.npz"
    if cache_file.exists():
        return _load_cache(cache_file)

    try:
        import demucs.api
    except ImportError as exc:
        raise RuntimeError(
            "demucs is not installed; install the analysis extra: pip install -e '.[analysis]'"
        ) from exc

    try:
        separator = demucs.api.Separator(model=MODEL_NAME, device=_device())
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Demucs model {MODEL_NAME!r} - weights may not be cached "
            f"locally and fetching them requires network access: {exc}"
        ) from exc

    if separator.samplerate != STEM_SR:
        raise RuntimeError(
            f"Demucs model {MODEL_NAME!r} reports samplerate {separator.samplerate}, "
            f"expected {STEM_SR}"
        )

    try:
        _origin, separated = separator.separate_audio_file(audio_path)
    except Exception as exc:
        raise RuntimeError(f"Demucs separation failed for {audio_path}: {exc}") from exc

    missing = [name for name in STEM_NAMES if name not in separated]
    if missing:
        raise RuntimeError(
            f"Demucs model {MODEL_NAME!r} did not produce stem(s) {missing}; "
            f"got {sorted(separated)}"
        )

    stems = {name: _to_mono_f32(separated[name]) for name in STEM_NAMES}

    try:
        np.savez(cache_file, **stems)
    except OSError as exc:
        print(f"warning: failed to cache stems to {cache_file}: {exc}", file=sys.stderr)

    return stems
