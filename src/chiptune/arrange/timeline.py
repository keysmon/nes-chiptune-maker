"""Frame-based channel representation.

The 2A03 is updated once per frame at 60 Hz, so a list of per-frame register
states is the natural representation - and it makes the "one note per channel"
hardware limit structurally true rather than something we have to assert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..score import Percussion


class ChannelId(str, Enum):
    PULSE1 = "pulse1"
    PULSE2 = "pulse2"
    TRIANGLE = "triangle"
    NOISE = "noise"


@dataclass(frozen=True)
class FrameEvent:
    """State of one channel during one frame. `pitch is None` means silent."""
    pitch: int | None = None
    volume: int = 0
    percussion: Percussion | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.volume <= 15:
            raise ValueError(f"volume must be 0-15 (4-bit DAC), got {self.volume}")

    @property
    def sounding(self) -> bool:
        return self.pitch is not None or self.percussion is not None


SILENT = FrameEvent()


@dataclass
class ChannelTimeline:
    channel: ChannelId
    frames: list[FrameEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)
