import numpy as np

from chiptune.analysis.tempo import estimate_grid


def test_estimates_known_click_tempo():
    sr = 22050
    bpm = 120
    dur = 8.0
    y = np.zeros(int(sr * dur), dtype=np.float32)
    step = int(sr * 60 / bpm)
    for i in range(0, len(y), step):
        y[i:i + 200] += np.hanning(min(200, len(y) - i))  # clicks
    grid = estimate_grid(y, sr)
    assert abs(grid.bpm - bpm) < 6 or abs(grid.bpm - bpm / 2) < 6 or abs(grid.bpm - bpm * 2) < 6
    assert grid.beats_per_bar == 4
