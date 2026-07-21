"""Hard invariants from spec 6.1.

These are the mechanical safety net that makes "listen only at the end"
survivable (Risk R1). They catch structural violations of the hardware model.
They explicitly do NOT catch a wrong bass octave, a buzzing arpeggio rate, or
drums that read as static - which is exactly why the end-of-project listen
still matters.
"""
from __future__ import annotations

import numpy as np

from .arrange.timeline import ChannelId, ChannelTimeline
from .nes.tables import playable_on_pulse, playable_on_triangle


class InvariantViolation(AssertionError):
    """A rendered arrangement violated a 2A03 hardware constraint."""


def check_invariants(
    timelines: dict[ChannelId, ChannelTimeline],
    samples: np.ndarray,
) -> None:
    # 1. The triangle channel has no volume control.
    tri = timelines[ChannelId.TRIANGLE]
    tri_volumes = {f.volume for f in tri.frames if f.pitch is not None}
    if len(tri_volumes) > 1:
        raise InvariantViolation(
            f"triangle volume varied across frames: {sorted(tri_volumes)}; "
            "the 2A03 triangle channel is on or off"
        )

    # 2. Every pitch must fit the 11-bit period register.
    for ch in (ChannelId.PULSE1, ChannelId.PULSE2):
        for i, f in enumerate(timelines[ch].frames):
            if f.pitch is not None and not playable_on_pulse(f.pitch):
                raise InvariantViolation(
                    f"{ch.value} frame {i}: pitch {f.pitch} has no valid 11-bit period"
                )
    for i, f in enumerate(tri.frames):
        if f.pitch is not None and not playable_on_triangle(f.pitch):
            raise InvariantViolation(
                f"triangle frame {i}: pitch {f.pitch} has no valid 11-bit period"
            )

    # 3. One note per channel per frame is structural (FrameEvent holds one pitch),
    #    so it is verified rather than assumed.
    for ch, tl in timelines.items():
        for i, f in enumerate(tl.frames):
            if f.pitch is not None and f.percussion is not None:
                raise InvariantViolation(
                    f"{ch.value} frame {i} carries both a pitch and a percussion hit"
                )

    # 4. No clipping.
    if samples.size and float(np.abs(samples).max()) > 1.0:
        raise InvariantViolation(
            f"output clips: peak {float(np.abs(samples).max()):.4f} exceeds 1.0"
        )
