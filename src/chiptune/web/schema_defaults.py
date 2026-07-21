"""Resolve a control's dot-path (e.g. "arrange.chord_octave", "drums.snare.volume",
"noise_lowpass_hz") to its current default value from the loaded default config."""
from __future__ import annotations

from ..config import default_raw_config


def current_default(path: str):
    raw = default_raw_config()
    node = raw
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
