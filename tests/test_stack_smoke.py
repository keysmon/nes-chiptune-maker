# tests/test_stack_smoke.py
"""Guards spec Risk R5: dependency fragility found only by execution."""
import numpy as np
import pytest


def test_core_imports():
    import chiptune  # noqa: F401
    import numpy, scipy, soundfile, pretty_midi  # noqa: F401


def test_setuptools_pin_holds():
    """resampy imports pkg_resources; setuptools>=81 removed it."""
    import setuptools
    major = int(setuptools.__version__.split(".")[0])
    assert major < 81, (
        f"setuptools {setuptools.__version__} removed pkg_resources; "
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
def test_torch_mps_available():
    import torch
    assert torch.backends.mps.is_available(), "Demucs separation expects MPS on this machine"
