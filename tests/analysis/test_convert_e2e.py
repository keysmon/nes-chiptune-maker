"""Slow end-to-end test: the real audio -> chiptune deliverable path.

This is the artifact the 2022 project set out to make. Runs the full pipeline
(separate -> transcribe -> assemble -> render) via the `convert` CLI on the real
pop clip and asserts a valid, song-length NES render comes out.
"""
from pathlib import Path

import pytest

SRC = Path("out/pop_src.wav")


@pytest.mark.slow
def test_convert_pop_song_produces_chiptune(tmp_path):
    if not SRC.exists():
        pytest.skip(f"{SRC} is missing; produced by the audio-analysis spike, not checked in")

    from chiptune.cli import main

    out = tmp_path / "pop_chiptune.wav"
    rc = main(["convert", str(SRC), "-o", str(out)])
    assert rc == 0 and out.exists()

    import soundfile as sf
    import numpy as np

    y, sr = sf.read(out)
    assert sr == 44100
    assert np.abs(y).max() > 0.05 and np.abs(y).max() <= 1.0
    # Song-length, not a single 1/60s frame: guards the score.duration wiring
    # that sizes the whole render.
    assert len(y) / sr > 5
