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


def test_symmetric_input_stays_zero_mean():
    """A zero-mean bipolar channel must stay zero-mean through the mixer.

    Our oscillators emit zero-mean band-limited waveforms. Half-wave rectifying
    one (clamping its negative half to 0) injects a positive DC offset and a
    buzzy octave-up artifact - worst on the triangle bass channel. The mixer's
    compression curve must be applied odd-symmetrically so a symmetric input
    produces a symmetric output.
    """
    tri = np.array([1.0, -1.0] * 50)
    z = np.zeros_like(tri)
    out = nes_mix(z, z, tri, z)
    assert abs(out.mean()) < 1e-9


def test_mixer_is_odd_symmetric():
    """nes_mix(-x) must equal -nes_mix(x): the DAC curve applies to both polarities."""
    rng = np.random.default_rng(0)
    p1 = rng.uniform(-7.0, 7.0, 100)
    p2 = rng.uniform(-7.0, 7.0, 100)
    tri = rng.uniform(-8.0, 8.0, 100)
    noise = rng.uniform(-1.0, 1.0, 100)
    pos = nes_mix(p1, p2, tri, noise)
    neg = nes_mix(-p1, -p2, -tri, -noise)
    np.testing.assert_allclose(neg, -pos, atol=1e-12)


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
