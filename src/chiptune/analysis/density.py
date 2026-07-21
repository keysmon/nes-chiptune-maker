"""Score density: how much is sounding, how often, per role.

The compass for "did the sparse arranger actually reduce clutter" -
`chiptune.arrange.chord_comp` replaces a wall of noisy transcribed HARMONY
notes with a clean comp, and `chiptune.arrange.sparse` thins LEAD/BASS
ornaments; this is the metric that turns "sounds less muddy" into a number
`build_score` can print alongside the artifact.
"""
from __future__ import annotations

import numpy as np

from ..score import Role, Score


def score_density(score: Score, frame_rate: float = 60.0) -> dict:
    """Rasterize `score.notes` onto a `frame_rate`-Hz grid over
    [0, score.duration) and measure clutter.

    Returns a dict with:
      - "per_role_active": {role: fraction of frames with >=1 note of that role}
      - "mean_simultaneous": mean, across all frames, of the count of
        simultaneously-sounding notes (all roles combined)
      - "notes_per_second": {role: note count / score.duration}
    An empty or zero-duration score returns all-zero values rather than
    raising (nothing to divide by, not an error).
    """
    per_role_active = {r: 0.0 for r in Role}
    notes_per_second = {r: 0.0 for r in Role}

    n_frames = max(0, round(score.duration * frame_rate))
    if n_frames == 0 or score.duration <= 0:
        return {
            "per_role_active": per_role_active,
            "mean_simultaneous": 0.0,
            "notes_per_second": notes_per_second,
        }

    # Sample frame centers, not edges, so a note that ends exactly on a frame
    # boundary doesn't get double-counted by the frame on either side of it.
    frame_times = (np.arange(n_frames) + 0.5) / frame_rate
    total_active = np.zeros(n_frames, dtype=np.int64)

    for role in Role:
        role_notes = score.notes_with_role(role)
        notes_per_second[role] = len(role_notes) / score.duration
        if not role_notes:
            continue
        role_active = np.zeros(n_frames, dtype=bool)
        for n in role_notes:
            mask = (frame_times >= n.start) & (frame_times < n.end)
            role_active |= mask
            total_active += mask
        per_role_active[role] = float(role_active.mean())

    return {
        "per_role_active": per_role_active,
        "mean_simultaneous": float(total_active.mean()),
        "notes_per_second": notes_per_second,
    }
