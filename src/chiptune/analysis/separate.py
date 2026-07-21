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
import os
import sys
import tempfile
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


def _save_cache_atomic(cache_file: Path, stems: dict[str, np.ndarray]) -> None:
    """Write the stem cache atomically.

    A partial write must never occupy the final cache path: a separation killed
    mid-write (Ctrl-C / OOM / disk-full) would otherwise leave a truncated
    `.npz` that wedges the input forever. So write to a temp file in the SAME
    directory (same filesystem, so the rename is atomic) then `os.replace` it
    over the final path. `np.savez` is handed the open file object, not a path,
    to avoid its habit of appending `.npz` to a bare temp name.
    """
    fd, tmp_name = tempfile.mkstemp(dir=cache_file.parent, prefix=".stemtmp-", suffix=".npz")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **stems)
        os.replace(tmp_path, cache_file)
    except OSError as exc:
        print(f"warning: failed to cache stems to {cache_file}: {exc}", file=sys.stderr)
    finally:
        # No-op after a successful replace; cleans the temp on any other path.
        tmp_path.unlink(missing_ok=True)


def _separate_uncached(audio_path: Path) -> dict[str, np.ndarray]:
    """Run Demucs on `audio_path`, returning mono float32 stems at STEM_SR.

    The single seam that touches the model; kept separate from caching so the
    cache logic is testable without a model load.
    """
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

    return {name: _to_mono_f32(separated[name]) for name in STEM_NAMES}


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
        try:
            return _load_cache(cache_file)
        except Exception as exc:  # noqa: BLE001 - "any load error" is intentional
            # A truncated/corrupt cache (BadZipFile / EOFError / KeyError /
            # ValueError / OSError) must self-heal, not wedge the input forever:
            # treat it as a miss, warn, and re-separate + overwrite atomically.
            print(
                f"warning: stem cache {cache_file} is unreadable "
                f"({type(exc).__name__}: {exc}); re-separating and overwriting",
                file=sys.stderr,
            )

    stems = _separate_uncached(audio_path)
    _save_cache_atomic(cache_file, stems)
    return stems
