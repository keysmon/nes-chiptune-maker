import numpy as np
import pytest

from chiptune.arrange.timeline import ChannelId, ChannelTimeline, FrameEvent, SILENT
from chiptune.invariants import InvariantViolation, check_invariants


def timelines(**overrides):
    base = {ch: ChannelTimeline(ch, [SILENT] * 10) for ch in ChannelId}
    base.update(overrides)
    return base


def test_clean_timelines_pass():
    check_invariants(timelines(), np.zeros(100))


def test_varying_triangle_volume_is_a_violation():
    tl = ChannelTimeline(ChannelId.TRIANGLE, [
        FrameEvent(pitch=40, volume=15),
        FrameEvent(pitch=40, volume=9),
    ])
    with pytest.raises(InvariantViolation, match="triangle volume"):
        check_invariants(timelines(**{ChannelId.TRIANGLE: tl}), np.zeros(100))


def test_out_of_range_pitch_is_a_violation():
    tl = ChannelTimeline(ChannelId.PULSE1, [FrameEvent(pitch=3, volume=8)])
    with pytest.raises(InvariantViolation, match="period"):
        check_invariants(timelines(**{ChannelId.PULSE1: tl}), np.zeros(100))


def test_clipping_is_a_violation():
    with pytest.raises(InvariantViolation, match="clip"):
        check_invariants(timelines(), np.array([0.0, 1.5, 0.0]))


def test_nan_sample_is_a_violation():
    """A NaN slips through both np.clip and the `nan > 1.0` clip comparison, so it
    must be caught explicitly - otherwise it reaches the WAV as silence-or-garbage."""
    with pytest.raises(InvariantViolation, match="finite"):
        check_invariants(timelines(), np.array([0.0, np.nan, 0.0]))


def test_volume_above_four_bits_is_rejected_at_construction():
    with pytest.raises(ValueError, match="0-15"):
        FrameEvent(pitch=60, volume=99)
