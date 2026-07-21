import numpy as np
import pytest

from chiptune.synth.pulse import PulseBank

SR = 44100


@pytest.fixture
def bank():
    return PulseBank(sample_rate=SR)


def test_output_length_matches_request(bank):
    out, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=1000, phase=0.0)
    assert out.shape == (1000,)


def test_output_is_zero_mean(bank):
    out, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=SR, phase=0.0)
    assert abs(out.mean()) < 0.02


def test_fundamental_lands_at_the_requested_frequency(bank):
    out, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=SR, phase=0.0)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / SR)[np.argmax(spec)]
    assert peak == pytest.approx(440.0, abs=2.0)


def test_fifty_percent_duty_suppresses_even_harmonics(bank):
    """A 50% square has only odd harmonics. This proves the table is a real pulse."""
    out, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=SR, phase=0.0)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)

    def energy_at(f):
        return spec[np.argmin(np.abs(freqs - f))]

    assert energy_at(880.0) < 0.05 * energy_at(440.0)   # 2nd harmonic suppressed
    assert energy_at(1320.0) > 0.15 * energy_at(440.0)  # 3rd harmonic present


def test_twenty_five_percent_duty_has_even_harmonics(bank):
    out, _ = bank.render(freq_hz=440.0, duty=0.25, n_samples=SR, phase=0.0)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)

    def energy_at(f):
        return spec[np.argmin(np.abs(freqs - f))]

    assert energy_at(880.0) > 0.2 * energy_at(440.0)


def test_no_aliasing_above_nyquist_for_a_high_note(bank):
    """Spec Risk R4. A naive square at 2 kHz would fold energy back below the fundamental."""
    out, _ = bank.render(freq_hz=2000.0, duty=0.5, n_samples=SR, phase=0.0)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    fundamental = spec[np.argmin(np.abs(freqs - 2000.0))]
    below = spec[freqs < 1900.0]
    assert below.max() < 0.05 * fundamental, "aliased energy folded below the fundamental"


def test_phase_is_continuous_across_calls(bank):
    a, ph = bank.render(freq_hz=440.0, duty=0.5, n_samples=500, phase=0.0)
    b, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=500, phase=ph)
    joined = np.concatenate([a, b])
    one_shot, _ = bank.render(freq_hz=440.0, duty=0.5, n_samples=1000, phase=0.0)
    np.testing.assert_allclose(joined, one_shot, atol=1e-9)


def test_amplitude_stays_within_unit_range(bank):
    for duty in (0.125, 0.25, 0.5, 0.75):
        out, _ = bank.render(freq_hz=440.0, duty=duty, n_samples=4410, phase=0.0)
        assert np.abs(out).max() <= 1.0 + 1e-6
