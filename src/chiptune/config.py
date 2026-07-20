"""Typed loader for config/nes.toml.

Every taste-sensitive value lives in TOML, never in Python. This is the
mitigation for spec Risk R1 (no listening loop until the end): a bad-sounding
result becomes a config edit rather than a rewrite.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "nes.toml"

VALID_DUTIES = (0.125, 0.25, 0.5, 0.75)
VALID_DRUM_KEYS = frozenset({"kick", "snare", "hat"})
VALID_LEVEL_KEYS = frozenset({"pulse1", "pulse2", "triangle", "noise"})


@dataclass(frozen=True)
class ChannelConfig:
    duty: float
    volume: int
    attack_frames: int
    decay_frames: int
    sustain: float
    release_frames: int

    def __post_init__(self) -> None:
        if not 0 <= self.volume <= 15:
            raise ValueError(f"volume must be 0-15 (4-bit DAC), got {self.volume}")
        if not 0.0 <= self.sustain <= 1.0:
            raise ValueError(f"sustain must be in [0, 1], got {self.sustain}")


@dataclass(frozen=True)
class PulseConfig(ChannelConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.duty not in VALID_DUTIES:
            raise ValueError(
                f"duty {self.duty} is not a real 2A03 value; must be one of {VALID_DUTIES}"
            )


@dataclass(frozen=True)
class ArrangeConfig:
    subdivision: int
    quantize_strength: float
    min_duration: float
    arpeggio_frames: int
    bass_low: int
    bass_high: int
    borrow_enabled: bool
    borrow_idle_frames: int
    borrow_hysteresis_frames: int

    def __post_init__(self) -> None:
        if self.bass_low >= self.bass_high:
            raise ValueError(f"bass_low {self.bass_low} must be below bass_high {self.bass_high}")
        if self.arpeggio_frames < 1:
            raise ValueError(f"arpeggio_frames must be >= 1, got {self.arpeggio_frames}")


@dataclass(frozen=True)
class DrumVoice:
    period_index: int
    mode: str
    volume: int
    frames: int

    def __post_init__(self) -> None:
        if not 0 <= self.period_index <= 15:
            raise ValueError(f"period_index must be 0-15, got {self.period_index}")
        if self.mode not in ("long", "short"):
            raise ValueError(f"mode must be 'long' or 'short', got {self.mode!r}")


@dataclass(frozen=True)
class Config:
    sample_rate: int
    frame_rate: float
    arrange: ArrangeConfig
    pulse1: PulseConfig
    pulse2: PulseConfig
    triangle: ChannelConfig
    noise: ChannelConfig
    drums: dict[str, DrumVoice] = field(default_factory=dict)
    levels: dict[str, float] = field(default_factory=dict)


_T = TypeVar("_T", bound=ChannelConfig)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    def channel(name: str, cls: type[_T] = ChannelConfig) -> _T:
        if name not in raw:
            raise ValueError(f"config {path} is missing required [{name}] section")
        return cls(**raw[name])

    raw_drums = raw.get("drums", {})
    bad_drums = set(raw_drums) - VALID_DRUM_KEYS
    if bad_drums:
        raise ValueError(
            f"config {path} has unknown [drums.{sorted(bad_drums)[0]}] section; "
            f"drums must be one of {sorted(VALID_DRUM_KEYS)}"
        )
    drums = {k: DrumVoice(**v) for k, v in raw_drums.items()}

    raw_levels = raw.get("levels", {})
    bad_levels = set(raw_levels) - VALID_LEVEL_KEYS
    if bad_levels:
        raise ValueError(
            f"config {path} has unknown [levels] key {sorted(bad_levels)[0]!r}; "
            f"levels must be one of {sorted(VALID_LEVEL_KEYS)}"
        )

    return Config(
        sample_rate=raw["sample_rate"],
        frame_rate=raw["frame_rate"],
        arrange=ArrangeConfig(**raw["arrange"]),
        pulse1=channel("pulse1", PulseConfig),
        pulse2=channel("pulse2", PulseConfig),
        triangle=channel("triangle"),
        noise=channel("noise"),
        drums=drums,
        levels=raw_levels,
    )
