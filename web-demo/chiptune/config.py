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
VALID_HARMONY_MODES = frozenset({"chords", "transcribe"})
# Must match the patterns chiptune.arrange.chord_comp.comp_chords understands.
VALID_CHORD_COMP_PATTERNS = frozenset({"up", "updown", "root_fifth"})
VALID_ARRANGE_KEYS = frozenset({
    "subdivision",
    "quantize_strength",
    "min_duration",
    "arpeggio_frames",
    "bass_low",
    "bass_high",
    "borrow_enabled",
    "borrow_idle_frames",
    "borrow_hysteresis_frames",
    "velocity_floor",
    "reattack_gap",
    "harmony_mode",
    "chord_comp_pattern",
    "chord_subdivision",
    "chord_octave",
    "chord_tones",
    "chord_smooth_beats",
    "melody_min_seconds",
    "bass_min_seconds",
    "harmony_rest_on_busy_melody",
})
VALID_ANALYSIS_KEYS = frozenset({
    "include_vocals",
    "vocal_fmin",
    "vocal_fmax",
    "min_note_seconds",
    "kick_band_hz",
    "hat_band_hz",
    "kick_low_frac_min",
    "hat_high_frac_min",
    "onset_backtrack",
    "harmony_declash",
    "declash_semitones",
})


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
    velocity_floor: float
    reattack_gap: float
    # Sparse arranger (chiptune.arrange.sparse / chord_comp / analysis.chords):
    # replaces basic-pitch-on-"other" HARMONY with a clean chord comp, and
    # thins LEAD/BASS ornaments - see chiptune.analysis.build_score.
    harmony_mode: str
    chord_comp_pattern: str
    chord_subdivision: int
    chord_octave: int
    chord_tones: int
    chord_smooth_beats: int
    melody_min_seconds: float
    bass_min_seconds: float
    harmony_rest_on_busy_melody: bool

    def __post_init__(self) -> None:
        if self.bass_low >= self.bass_high:
            raise ValueError(f"bass_low {self.bass_low} must be below bass_high {self.bass_high}")
        if self.arpeggio_frames < 1:
            raise ValueError(f"arpeggio_frames must be >= 1, got {self.arpeggio_frames}")
        if not 0.0 <= self.velocity_floor <= 1.0:
            raise ValueError(f"velocity_floor must be in [0, 1], got {self.velocity_floor}")
        if self.reattack_gap < 0:
            raise ValueError(f"reattack_gap must be >= 0, got {self.reattack_gap}")
        if self.harmony_mode not in VALID_HARMONY_MODES:
            raise ValueError(
                f"harmony_mode {self.harmony_mode!r} must be one of {sorted(VALID_HARMONY_MODES)}"
            )
        if self.chord_comp_pattern not in VALID_CHORD_COMP_PATTERNS:
            raise ValueError(
                f"chord_comp_pattern {self.chord_comp_pattern!r} must be one of "
                f"{sorted(VALID_CHORD_COMP_PATTERNS)}"
            )
        if not 1 <= self.chord_subdivision <= 16:
            raise ValueError(f"chord_subdivision must be 1-16, got {self.chord_subdivision}")
        if self.chord_tones < 1:
            raise ValueError(f"chord_tones must be >= 1, got {self.chord_tones}")
        if not 0 <= self.chord_octave <= 8:
            raise ValueError(f"chord_octave must be 0-8, got {self.chord_octave}")
        if self.chord_smooth_beats < 1:
            raise ValueError(f"chord_smooth_beats must be >= 1, got {self.chord_smooth_beats}")
        if self.melody_min_seconds < 0:
            raise ValueError(f"melody_min_seconds must be >= 0, got {self.melody_min_seconds}")
        if self.bass_min_seconds < 0:
            raise ValueError(f"bass_min_seconds must be >= 0, got {self.bass_min_seconds}")


@dataclass(frozen=True)
class VibratoConfig:
    rate_hz: float
    depth_semitones: float
    delay_frames: int
    enabled: bool

    def __post_init__(self) -> None:
        if self.rate_hz < 0:
            raise ValueError(f"rate_hz must be >= 0, got {self.rate_hz}")
        if self.depth_semitones < 0:
            raise ValueError(f"depth_semitones must be >= 0, got {self.depth_semitones}")
        if self.delay_frames < 0:
            raise ValueError(f"delay_frames must be >= 0, got {self.delay_frames}")


@dataclass(frozen=True)
class DrumVoice:
    period_index: int
    mode: str
    volume: int
    frames: int
    priority: int  # collision winner when two hits land on the same frame; higher wins

    def __post_init__(self) -> None:
        if not 0 <= self.period_index <= 15:
            raise ValueError(f"period_index must be 0-15, got {self.period_index}")
        if self.mode not in ("long", "short"):
            raise ValueError(f"mode must be 'long' or 'short', got {self.mode!r}")
        if not 0 <= self.volume <= 15:
            raise ValueError(f"volume must be 0-15 (4-bit DAC), got {self.volume}")
        if self.frames < 1:
            raise ValueError(f"frames must be >= 1, got {self.frames}")


@dataclass(frozen=True)
class AnalysisConfig:
    include_vocals: bool
    vocal_fmin: float
    vocal_fmax: float
    min_note_seconds: float
    kick_band_hz: float
    hat_band_hz: float
    kick_low_frac_min: float
    hat_high_frac_min: float
    onset_backtrack: bool
    harmony_declash: bool
    declash_semitones: int

    def __post_init__(self) -> None:
        if self.vocal_fmin >= self.vocal_fmax:
            raise ValueError(
                f"vocal_fmin {self.vocal_fmin} must be below vocal_fmax {self.vocal_fmax}"
            )
        if self.kick_band_hz >= self.hat_band_hz:
            raise ValueError(
                f"kick_band_hz {self.kick_band_hz} must be below hat_band_hz {self.hat_band_hz}"
            )
        if not 0 < self.kick_low_frac_min <= 1:
            raise ValueError(
                f"kick_low_frac_min must be in (0, 1], got {self.kick_low_frac_min}"
            )
        if not 0 < self.hat_high_frac_min <= 1:
            raise ValueError(
                f"hat_high_frac_min must be in (0, 1], got {self.hat_high_frac_min}"
            )
        if self.min_note_seconds <= 0:
            raise ValueError(f"min_note_seconds must be positive, got {self.min_note_seconds}")


@dataclass(frozen=True)
class Config:
    sample_rate: int
    frame_rate: float
    arrange: ArrangeConfig
    pulse1: PulseConfig
    pulse2: PulseConfig
    triangle: ChannelConfig
    noise: ChannelConfig
    analysis: AnalysisConfig
    vibrato: VibratoConfig
    noise_lowpass_hz: float = 0.0  # low-pass the noise channel to tame harsh/hissy drums; 0 = off
    output_highpass_hz: float = 0.0  # sub-sonic rumble cut on the final mix; 0 = off
    output_lowpass_hz: float = 0.0   # gentle top-end rolloff on the final mix; 0 = off
    drums: dict[str, DrumVoice] = field(default_factory=dict)
    levels: dict[str, float] = field(default_factory=dict)


_T = TypeVar("_T", bound=ChannelConfig)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return config_from_dict(raw, source=str(path))


def default_raw_config() -> dict:
    """The default config as a plain nested dict (parsed from nes.toml). The web
    runtime deep-merges user overrides onto this before building a Config."""
    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def config_from_dict(raw: dict, source: str = "config") -> Config:
    """Build (and validate) a Config from a raw nested dict. Shared by load_config
    (file) and the web runtime (defaults + JSON overrides)."""
    def channel(name: str, cls: type[_T]) -> _T:
        if name not in raw:
            raise ValueError(f"config {source} is missing required [{name}] section")
        return cls(**raw[name])
    path = source  # keep the existing error messages below unchanged

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

    if "arrange" not in raw:
        raise ValueError(f"config {path} is missing required [arrange] section")
    bad_arrange = set(raw["arrange"]) - VALID_ARRANGE_KEYS
    if bad_arrange:
        raise ValueError(
            f"config {path} has unknown [arrange] key {sorted(bad_arrange)[0]!r}; "
            f"arrange must be one of {sorted(VALID_ARRANGE_KEYS)}"
        )

    if "analysis" not in raw:
        raise ValueError(f"config {path} is missing required [analysis] section")
    raw_analysis = raw["analysis"]
    bad_analysis = set(raw_analysis) - VALID_ANALYSIS_KEYS
    if bad_analysis:
        raise ValueError(
            f"config {path} has unknown [analysis] key {sorted(bad_analysis)[0]!r}; "
            f"analysis must be one of {sorted(VALID_ANALYSIS_KEYS)}"
        )

    if "vibrato" not in raw:
        raise ValueError(f"config {path} is missing required [vibrato] section")

    return Config(
        sample_rate=raw["sample_rate"],
        frame_rate=raw["frame_rate"],
        arrange=ArrangeConfig(**raw["arrange"]),
        pulse1=channel("pulse1", PulseConfig),
        pulse2=channel("pulse2", PulseConfig),
        triangle=channel("triangle", ChannelConfig),
        noise=channel("noise", ChannelConfig),
        analysis=AnalysisConfig(**raw_analysis),
        vibrato=VibratoConfig(**raw["vibrato"]),
        noise_lowpass_hz=raw.get("noise_lowpass_hz", 0.0),
        output_highpass_hz=raw.get("output_highpass_hz", 0.0),
        output_lowpass_hz=raw.get("output_lowpass_hz", 0.0),
        drums=drums,
        levels=raw_levels,
    )


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base` (nested dicts merge;
    scalars replace). Used to layer web-UI overrides onto the default config."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_from_overrides(overrides: dict | None) -> Config:
    """Build a validated Config from the default nes.toml deep-merged with a nested
    `overrides` dict (e.g. {"levels": {"noise": 0.5}, "vibrato": {"depth_semitones": 0.5}}).
    Invalid values raise ValueError via the dataclass validators - the web API turns
    that into a 400. This is the single entry point for runtime (web) configuration."""
    merged = _deep_merge(default_raw_config(), overrides or {})
    return config_from_dict(merged, source="overrides")
