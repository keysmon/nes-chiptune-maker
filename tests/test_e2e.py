"""End-to-end: the real CLI invocation, which is what 'done' requires.

Passing unit tests alone does not constitute done (spec 7).
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from chiptune.cli import main, render_midi

ASSET = Path(__file__).resolve().parents[1] / "assets" / "test_theme.mid"


def test_cli_renders_the_test_theme(tmp_path):
    out = tmp_path / "theme.wav"
    rc = main(["render", str(ASSET), "-o", str(out)])
    assert rc == 0
    assert out.exists()

    audio, sr = sf.read(out)
    assert sr == 44100
    assert len(audio) / sr > 15.0, "test theme is 16 s of music"
    assert np.abs(audio).max() > 0.05, "output is audibly silent"
    assert np.abs(audio).max() <= 1.0, "output clips"


def test_render_is_byte_deterministic(tmp_path):
    """Spec 6.1, scoped to Score -> wav only."""
    a = render_midi(ASSET, out_path=tmp_path / "a.wav")
    b = render_midi(ASSET, out_path=tmp_path / "b.wav")
    assert a.read_bytes() == b.read_bytes()


def test_cli_reports_missing_input(tmp_path, capsys):
    rc = main(["render", str(tmp_path / "nope.mid"), "-o", str(tmp_path / "x.wav")])
    assert rc != 0
    assert "not found" in capsys.readouterr().err.lower()
