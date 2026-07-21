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


def test_fundamental_amplitude_is_consistent_across_octaves(bank):
    """A note's perceived loudness is its fundamental, and a pulse's fundamental
    amplitude (2/pi)*sin(pi*duty) does NOT depend on octave. The mip pyramid must
    preserve that: high-octave tables legitimately keep fewer harmonics (a thinner
    tone - authentic NES band-limiting), but the fundamental must stay put.

    Normalizing each table to its OWN peak would divide the sparse high-octave
    tables by a smaller number and scale their fundamental up, inverting the
    loudness by ~12 dB for narrow duties. A single global gain keeps the
    fundamental flat. This asserts the fundamental, NOT overall peak or RMS - those
    legitimately differ across octaves because high notes carry fewer harmonics.

    Duty 0.125 is the stress case, and the probe MIDIs must reach the top octaves
    where the harmonic count collapses: 48/72/96/120 land in octave tables 3/5/7/9.
    (The suggested 48/60/72 all sit in octaves 3/4/5, where every table still holds
    dozens of harmonics and per-table peaks are near-equal - so that set passes
    even with the inverting bug and would test nothing.)
    """
    duty = 0.125
    mags = []
    for midi in (48, 72, 96, 120):
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        out, _ = bank.render(freq_hz=freq, duty=duty, n_samples=SR, phase=0.0)
        spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1 / SR)
        mags.append(float(spec[np.argmin(np.abs(freqs - freq))]))
    assert (max(mags) - min(mags)) / max(mags) < 0.10, \
        f"fundamental amplitude varies across octaves: {mags}"
