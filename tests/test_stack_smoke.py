# tests/test_stack_smoke.py
"""Guards spec Risk R5: dependency fragility found only by execution.

Supported target platform: arm64 macOS (Apple Silicon). demucs's dependency
markers resolve a different stack on x86_64 Darwin (numpy<2, torch<2.3), so
tests that assume the arm64 resolution (e.g. MPS availability) skip cleanly
elsewhere instead of asserting a stack this suite never installs.
"""
import platform

import numpy as np
import pytest

_IS_ARM64_MACOS = platform.system() == "Darwin" and platform.machine() == "arm64"


def test_core_imports():
    import chiptune  # noqa: F401
    import numpy, scipy, soundfile, pretty_midi, mido  # noqa: F401


def test_setuptools_pin_holds():
    """resampy imports pkg_resources; setuptools>=81 removed it.

    Import the module directly rather than checking the version string, since
    the version check can drift from the actual failure mode (a version bump
    that keeps pkg_resources, or a repackaging that drops it early).
    """
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        pytest.fail(
            "pkg_resources is unavailable; setuptools>=81 removed it and "
            "basic-pitch will fail to import. Reinstall with -c constraints.txt"
        )


@pytest.mark.slow
def test_basic_pitch_imports_and_transcribes(tmp_path):
    """Phase 2 depends on this. Verified working 2026-07-20 via CoreML backend."""
    import soundfile as sf
    from basic_pitch.inference import predict

    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sig = np.concatenate([
        0.5 * np.sin(2 * np.pi * 440.0 * t),      # A4  -> MIDI 69
        0.5 * np.sin(2 * np.pi * 523.25 * t),     # C5  -> MIDI 72
    ]).astype(np.float32)
    wav = tmp_path / "smoke.wav"
    sf.write(wav, sig, sr)

    _, _, notes = predict(str(wav))
    pitches = sorted({p for _, _, p, _, _ in notes})
    assert 69 in pitches and 72 in pitches


@pytest.mark.slow
def test_demucs_separates_synthetic_signal_into_four_stems():
    """Runs a real separation via the Python API, not just an import.

    Demucs is the highest-risk dependency in this stack (largest model,
    slowest load, most native extensions), so import-only coverage would
    miss exactly the fragility this suite exists to catch. Uses the API
    directly, never a shell-out, so failures surface as exceptions.
    """
    import torch
    from demucs.api import Separator

    separator = Separator(model="htdemucs", progress=False)
    sr = separator.samplerate
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    left = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.1 * np.sin(2 * np.pi * 330.0 * t)
    wav = torch.from_numpy(np.stack([left, right]).astype(np.float32))

    _, stems = separator.separate_tensor(wav, sr=sr)

    assert set(stems.keys()) == {"drums", "bass", "other", "vocals"}


@pytest.mark.slow
@pytest.mark.skipif(not _IS_ARM64_MACOS, reason="MPS is only expected on arm64 macOS")
def test_torch_mps_available():
    import torch
    assert torch.backends.mps.is_available(), "Demucs separation expects MPS on this machine"
