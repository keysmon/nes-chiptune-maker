"""The set of config knobs the web UI exposes, as data. Each control names the
nested config path it overrides, its type/range, and a group for layout. The
frontend builds the panel from this; the backend applies the override dict.

`path` is a dot-path into the config-override dict, e.g. "arrange.chord_octave"
becomes {"arrange": {"chord_octave": <value>}}. Drum paths use
"drums.snare.volume" -> {"drums": {"snare": {"volume": <value>}}}.
"""
from __future__ import annotations

from .schema_defaults import current_default

# group, then controls. `type`: range | toggle | choice.
CONTROLS = [
    # --- Harmony (analysis; changing these re-analyzes) ---
    {"group": "Harmony", "path": "arrange.harmony_mode", "label": "Harmony source",
     "type": "choice", "options": ["chords", "transcribe"], "analysis": True,
     "help": "chords = detect the progression and comp it cleanly; transcribe = the raw (muddier) note-for-note harmony"},
    {"group": "Harmony", "path": "arrange.chord_octave", "label": "Chord octave",
     "type": "range", "min": 2, "max": 6, "step": 1, "analysis": True},
    {"group": "Harmony", "path": "arrange.chord_comp_pattern", "label": "Comp pattern",
     "type": "choice", "options": ["up", "updown", "root_fifth"], "analysis": True},
    {"group": "Harmony", "path": "arrange.chord_subdivision", "label": "Comp speed (notes/beat)",
     "type": "range", "min": 1, "max": 8, "step": 1, "analysis": True},
    {"group": "Harmony", "path": "arrange.chord_tones", "label": "Chord tones",
     "type": "range", "min": 2, "max": 3, "step": 1, "analysis": True},
    {"group": "Harmony", "path": "arrange.harmony_rest_on_busy_melody", "label": "Rest under melody",
     "type": "toggle", "analysis": True, "help": "clear harmony on melodic attacks for space"},
    {"group": "Harmony", "path": "analysis.harmony_declash", "label": "De-clash vs lead",
     "type": "toggle", "analysis": True},

    # --- Arrangement (analysis) ---
    {"group": "Arrangement", "path": "analysis.include_vocals", "label": "Vocal lead",
     "type": "toggle", "analysis": True, "help": "off = instrumental cover, lead from the 'other' stem"},
    {"group": "Arrangement", "path": "arrange.melody_min_seconds", "label": "Melody thinning",
     "type": "range", "min": 0.0, "max": 0.3, "step": 0.01, "analysis": True},
    {"group": "Arrangement", "path": "arrange.bass_min_seconds", "label": "Bass thinning",
     "type": "range", "min": 0.0, "max": 0.3, "step": 0.01, "analysis": True},

    # --- Feel (synthesis) ---
    {"group": "Feel", "path": "arrange.quantize_strength", "label": "Timing tightness",
     "type": "range", "min": 0.0, "max": 1.0, "step": 0.05},
    {"group": "Feel", "path": "arrange.arpeggio_frames", "label": "Arpeggio rate (frames)",
     "type": "range", "min": 1, "max": 6, "step": 1},
    {"group": "Feel", "path": "arrange.velocity_floor", "label": "Dynamics range",
     "type": "range", "min": 0.1, "max": 1.0, "step": 0.05,
     "help": "lower = more soft/loud contrast"},

    # --- Vibrato (synthesis) ---
    {"group": "Vibrato", "path": "vibrato.enabled", "label": "Vibrato", "type": "toggle"},
    {"group": "Vibrato", "path": "vibrato.rate_hz", "label": "Rate (Hz)",
     "type": "range", "min": 0.0, "max": 10.0, "step": 0.5},
    {"group": "Vibrato", "path": "vibrato.depth_semitones", "label": "Depth (semitones)",
     "type": "range", "min": 0.0, "max": 1.0, "step": 0.05},

    # --- Levels (synthesis) ---
    {"group": "Levels", "path": "levels.pulse1", "label": "Lead", "type": "range", "min": 0.0, "max": 1.5, "step": 0.05},
    {"group": "Levels", "path": "levels.pulse2", "label": "Harmony", "type": "range", "min": 0.0, "max": 1.5, "step": 0.05},
    {"group": "Levels", "path": "levels.triangle", "label": "Bass", "type": "range", "min": 0.0, "max": 1.5, "step": 0.05},
    {"group": "Levels", "path": "levels.noise", "label": "Drums", "type": "range", "min": 0.0, "max": 1.5, "step": 0.05},

    # --- Drums (synthesis) ---
    {"group": "Drums", "path": "drums.kick.volume", "label": "Kick vol", "type": "range", "min": 0, "max": 15, "step": 1},
    {"group": "Drums", "path": "drums.kick.period_index", "label": "Kick pitch", "type": "range", "min": 0, "max": 15, "step": 1},
    {"group": "Drums", "path": "drums.snare.volume", "label": "Snare vol", "type": "range", "min": 0, "max": 15, "step": 1},
    {"group": "Drums", "path": "drums.hat.volume", "label": "Hat vol", "type": "range", "min": 0, "max": 15, "step": 1},
    {"group": "Drums", "path": "noise_lowpass_hz", "label": "Drum tone (low-pass Hz)",
     "type": "range", "min": 1000, "max": 12000, "step": 250},

    # --- Output (synthesis) ---
    {"group": "Output", "path": "output_lowpass_hz", "label": "Warmth (low-pass Hz)",
     "type": "range", "min": 6000, "max": 20000, "step": 500},
]


def schema_with_defaults() -> list[dict]:
    """Attach each control's current default value (from nes.toml) for the UI."""
    out = []
    for c in CONTROLS:
        d = dict(c)
        d["default"] = current_default(c["path"])
        d.setdefault("analysis", False)
        out.append(d)
    return out
