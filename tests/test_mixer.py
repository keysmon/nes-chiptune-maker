import numpy as np
import pytest

from chiptune.synth.mixer import nes_mix, write_wav


def const(v, n=100):
    return np.full(n, float(v))


def test_silence_in_silence_out():
    out = nes_mix(const(0), const(0), const(0), const(0))
    np.testing.assert_allclose(out, 0.0)


def test_mixing_is_non_linear_not_additive():
    """Two pulses at full tilt must be quieter than twice one pulse."""
    one = nes_mix(const(15), const(0), const(0), const(0))[0]
    two = nes_mix(const(15), const(15), const(0), const(0))[0]
    assert two < 2 * one


def test_output_never_clips():
    out = nes_mix(const(15), const(15), const(15), const(15))
    assert np.abs(out).max() <= 1.0


def test_louder_input_gives_louder_output():
    quiet = nes_mix(const(4), const(0), const(0), const(0))[0]
    loud = nes_mix(const(12), const(0), const(0), const(0))[0]
    assert loud > quiet


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="same length"):
        nes_mix(const(1, 100), const(1, 50), const(1, 100), const(1, 100))


def test_write_wav_round_trips(tmp_path):
    import soundfile as sf
    path = tmp_path / "out.wav"
    sig = np.sin(np.linspace(0, 10, 4410))
    write_wav(path, sig, 44100)
    back, sr = sf.read(path)
    assert sr == 44100
    np.testing.assert_allclose(back, sig, atol=1e-4)
