import numpy as np
import pytest

from chiptune.synth.noise import lfsr_sequence, render_noise

SR = 44100


def test_long_mode_sequence_is_the_full_lfsr_period():
    assert lfsr_sequence("long").shape == (32767,)


def test_short_mode_sequence_is_93_steps():
    """Mode 1 taps bit 6 instead of bit 1, giving a 93-step tonal buzz."""
    assert lfsr_sequence("short").shape == (93,)


def test_sequence_values_are_bipolar():
    assert set(np.unique(lfsr_sequence("long"))) == {-1.0, 1.0}


def test_render_length_matches_request():
    out, _ = render_noise(period_index=8, mode="long", n_samples=1000,
                          sample_rate=SR, cursor=0)
    assert out.shape == (1000,)


def test_lower_period_index_is_brighter():
    """Period index selects the LFSR clock rate; smaller period = higher rate."""
    fast, _ = render_noise(0, "long", SR, SR, 0)
    slow, _ = render_noise(15, "long", SR, SR, 0)

    def centroid(x):
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        return float((spec * freqs).sum() / spec.sum())

    assert centroid(fast) > centroid(slow)


def test_cursor_advances_so_repeated_calls_do_not_repeat_audio():
    a, cur = render_noise(8, "long", 1000, SR, 0)
    b, _ = render_noise(8, "long", 1000, SR, cur)
    assert not np.allclose(a, b)


def test_rejects_bad_period_index():
    with pytest.raises(ValueError, match="period_index"):
        render_noise(16, "long", 100, SR, 0)


def test_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        render_noise(0, "medium", 100, SR, 0)
