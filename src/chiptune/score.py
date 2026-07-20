# src/chiptune/score.py
"""Chip-agnostic musical score.

This is the seam between the analysis half (separation, transcription) and the
realization half (arrangement, synthesis). It carries a tempo backbone rather
than free-floating events, because un-quantized pitch and timing sound unstable
on a pulse channel and fight the 60 Hz frame clock.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np


class Role(str, Enum):
    LEAD = "lead"
    HARMONY = "harmony"
    BASS = "bass"
    PERCUSSION = "percussion"


class Percussion(str, Enum):
    KICK = "kick"
    SNARE = "snare"
    HAT = "hat"


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int
    role: Role
    percussion: Percussion | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end must be after start (got {self.start} -> {self.end})")
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch must be MIDI 0-127 (got {self.pitch})")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity must be 0-127 (got {self.velocity})")
        if self.role is Role.PERCUSSION and self.percussion is None:
            raise ValueError("PERCUSSION notes require a `percussion` kind")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TempoGrid:
    bpm: float
    offset: float
    beats_per_bar: int

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError(f"bpm must be positive (got {self.bpm})")

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    def subdivision_times(self, subdivision: int, duration: float) -> np.ndarray:
        """Grid line times in seconds. `subdivision` is in note values: 16 == 1/16 notes."""
        if subdivision <= 0:
            raise ValueError(f"subdivision must be positive (got {subdivision})")
        step = self.seconds_per_beat * 4.0 / subdivision
        n = int(np.ceil((duration - self.offset) / step))
        return self.offset + step * np.arange(max(n, 0))


@dataclass
class Score:
    tempo: TempoGrid
    notes: list[NoteEvent] = field(default_factory=list)
    duration: float = 0.0

    def notes_with_role(self, role: Role) -> list[NoteEvent]:
        return [n for n in self.notes if n.role is role]

    def to_json(self) -> str:
        return json.dumps(
            {
                "tempo": asdict(self.tempo),
                "duration": self.duration,
                "notes": [
                    {
                        "pitch": n.pitch,
                        "start": n.start,
                        "end": n.end,
                        "velocity": n.velocity,
                        "role": n.role.value,
                        "percussion": n.percussion.value if n.percussion else None,
                    }
                    for n in self.notes
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "Score":
        raw = json.loads(s)
        return cls(
            tempo=TempoGrid(**raw["tempo"]),
            duration=raw["duration"],
            notes=[
                NoteEvent(
                    pitch=n["pitch"],
                    start=n["start"],
                    end=n["end"],
                    velocity=n["velocity"],
                    role=Role(n["role"]),
                    percussion=Percussion(n["percussion"]) if n["percussion"] else None,
                )
                for n in raw["notes"]
            ],
        )
