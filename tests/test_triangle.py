import numpy as np
import pytest

from chiptune.synth.triangle import TRIANGLE_TABLE, render_triangle

SR = 44100


def test_table_is_a_32_step_staircase():
    assert TRIANGLE_TABLE.shape == (32,)
    assert TRIANGLE_TABLE.min() == pytest.approx(-1.0)
    assert TRIANGLE_TABLE.max() == pytest.approx(1.0)


def test_table_rises_then_falls():
    first, second = TRIANGLE_TABLE[:16], TRIANGLE_TABLE[16:]
    assert np.all(np.diff(first) > 0)
    assert np.all(np.diff(second) < 0)


def test_only_sixteen_distinct_levels():
    """4-bit DAC: the staircase has 16 levels, each visited twice."""
    assert len(np.unique(np.round(TRIANGLE_TABLE, 6))) == 16


def test_fundamental_lands_at_the_requested_frequency():
    out, _ = render_triangle(220.0, SR, SR, 0.0)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / SR)[np.argmax(spec)]
    assert peak == pytest.approx(220.0, abs=2.0)


def test_phase_is_continuous_across_calls():
    a, ph = render_triangle(220.0, 500, SR, 0.0)
    b, _ = render_triangle(220.0, 500, SR, ph)
    one_shot, _ = render_triangle(220.0, 1000, SR, 0.0)
    np.testing.assert_allclose(np.concatenate([a, b]), one_shot, atol=1e-9)
