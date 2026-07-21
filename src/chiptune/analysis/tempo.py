"""Tempo / beat-grid estimation.

The realization half needs a `TempoGrid` (bpm + first-beat offset) to quantize
onsets against, not a free-running click count - see `chiptune.score.TempoGrid`.
"""
from __future__ import annotations

import sys

import numpy as np
import librosa

from chiptune.score import TempoGrid


def estimate_grid(mono: np.ndarray, sr: int, beats_per_bar: int = 4) -> TempoGrid:
    """Estimate a `TempoGrid` from a mono audio signal via beat tracking.

    Octave errors (half/double the true tempo) are a known beat-tracking
    failure mode - see spec Risk R6 - and are not corrected here.
    """
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr)
    if len(beats) == 0:
        # `beat_track`'s internal onset envelope uses median aggregation
        # across mel bands, which degenerates to all-zero (and therefore
        # 0 BPM / no beats - a documented librosa failure mode) for sharp,
        # narrowband-in-time transients such as a click track, even though
        # harmonically dense real music tracks fine with it. Retry with a
        # mean-aggregated envelope - librosa.onset.onset_strength's own
        # default - which does not degenerate on this kind of input.
        #
        # This fallback only fires on the degenerate (no-beats) case, never
        # unconditionally: mean vs. median aggregation lock onto different
        # metrical levels on real music (verified 89 BPM w/ median vs.
        # 136 BPM w/ mean on the same pop clip - a ~3:2 ratio, not the 2x
        # octave error the caller already tolerates), so always preferring
        # mean would silently change tempo estimates on real audio.
        print(
            "chiptune.analysis.tempo: median-aggregated onset envelope produced no "
            "beats; retrying with mean aggregation",
            file=sys.stderr,
        )
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0:
        raise RuntimeError("librosa could not detect any beats in the given audio")
    offset = float(librosa.frames_to_time(beats[0], sr=sr)) if len(beats) > 0 else 0.0
    return TempoGrid(bpm=bpm, offset=offset, beats_per_bar=beats_per_bar)
