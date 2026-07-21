"""Beat-synchronous chord progression detection.

The muddy-harmony problem this replaces: basic-pitch transcribing the "other"
stem note-by-note produces hundreds of near-simultaneous noisy notes. A chord
progression is a much smaller, cleaner description of the same harmonic
content - one label per beat, not dozens of overlapping pitches - and is what
`chiptune.arrange.chord_comp` comps into clean arpeggios.

Pipeline: chroma_cqt gives 12-bin pitch-class energy per frame; pool it to
the beat grid from `TempoGrid` (so labels land on musically meaningful
boundaries instead of arbitrary frame windows); match each beat's pooled
chroma against 24 triad templates (12 major, 12 minor) by dot product; mode-
smooth the label sequence to drop isolated single-beat misreads; merge
consecutive equal labels into `ChordSegment`s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from ..score import TempoGrid

# Triad intervals in semitones above the root.
_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)


@dataclass(frozen=True)
class ChordSegment:
    start: float
    end: float
    root: int  # pitch class 0-11; 0 = C, matching librosa chroma bin 0
    is_minor: bool


def _triad_templates() -> tuple[np.ndarray, list[tuple[int, bool]]]:
    """24 triad templates (12 major roots, then 12 minor roots), each a 12-vector.

    `templates[i]` is the chord template for `labels[i] = (root, is_minor)`.
    A template has 1.0 at each chord-tone pitch class (root's intervals
    rotated by the root) and 0 elsewhere - unnormalized, since all templates
    have equal norm (3 chord tones) so dot product ranks them the same as
    cosine similarity would.
    """
    templates = []
    labels: list[tuple[int, bool]] = []
    for root in range(12):
        for intervals, is_minor in ((_MAJOR_INTERVALS, False), (_MINOR_INTERVALS, True)):
            vec = np.zeros(12, dtype=np.float64)
            for interval in intervals:
                vec[(root + interval) % 12] = 1.0
            templates.append(vec)
            labels.append((root, is_minor))
    return np.stack(templates), labels


def _beat_times(grid: TempoGrid, duration: float) -> np.ndarray:
    """Beat times implied by `grid` covering [grid.offset, duration)."""
    if duration <= grid.offset:
        return np.array([grid.offset])
    n = int(np.floor((duration - grid.offset) / grid.seconds_per_beat)) + 1
    return grid.offset + grid.seconds_per_beat * np.arange(n, dtype=np.float64)


def _mode_smooth(labels: list[int], window: int) -> list[int]:
    """Replace each label with the most common label in a centered window.

    A mode filter, not a numeric median: labels are categorical (template
    index 0-23) with no meaningful order, so a median over that encoding
    could select a value that does not even occur in the window. Mode is
    the categorical analogue and works for any window size, including even
    ones, where a true median has no single well-defined value.
    """
    if window <= 1:
        return list(labels)
    n = len(labels)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts: dict[int, int] = {}
        for lab in labels[lo:hi]:
            counts[lab] = counts.get(lab, 0) + 1
        # Ties favor the center label itself (least disruptive to the run
        # it's already part of) so a flat 50/50 window at a chord boundary
        # doesn't flip-flop.
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == labels[i]))[0]
        out.append(best)
    return out


def detect_chords(
    mono: np.ndarray, sr: int, grid: TempoGrid, smooth_beats: int = 2
) -> list[ChordSegment]:
    """Detect the chord progression of `mono` on `grid`'s beat grid."""
    duration = len(mono) / sr
    # Too short to hold a chord: a micro-clip yields a spurious single garbage
    # chord (librosa pads to >=1 frame), so return nothing rather than a wrong label.
    if mono.size == 0 or duration < grid.seconds_per_beat:
        return []

    # Both librosa calls must use the same hop or the frame->time mapping drifts;
    # bind it once so a future edit can't desync them silently.
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop)  # (12, n_frames)
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return []
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    beats = _beat_times(grid, duration)
    templates, labels = _triad_templates()

    beat_label_idx: list[int] = []
    beat_bounds: list[tuple[float, float]] = []
    for i, raw_beat_start in enumerate(beats):
        beat_start = float(raw_beat_start)
        beat_end = float(beats[i + 1]) if i + 1 < len(beats) else duration
        if beat_end <= beat_start:
            continue
        mask = (frame_times >= beat_start) & (frame_times < beat_end)
        if not np.any(mask):
            # Beat window narrower than one chroma hop (e.g. a fast tempo
            # against a long CQT hop): fall back to the nearest frame so a
            # short beat doesn't silently vanish from the label sequence.
            idx = int(np.argmin(np.abs(frame_times - beat_start)))
            mask = np.zeros(n_frames, dtype=bool)
            mask[idx] = True
        pooled = chroma[:, mask].mean(axis=1)
        scores = templates @ pooled
        beat_label_idx.append(int(np.argmax(scores)))
        beat_bounds.append((beat_start, beat_end))

    if not beat_label_idx:
        return []

    smoothed = _mode_smooth(beat_label_idx, smooth_beats)

    segments: list[ChordSegment] = []
    seg_start, seg_end = beat_bounds[0]
    seg_label = smoothed[0]
    for (b_start, b_end), lab in zip(beat_bounds[1:], smoothed[1:]):
        if lab == seg_label:
            seg_end = b_end
        else:
            root, is_minor = labels[seg_label]
            segments.append(ChordSegment(seg_start, seg_end, root, is_minor))
            seg_start, seg_end, seg_label = b_start, b_end, lab
    root, is_minor = labels[seg_label]
    segments.append(ChordSegment(seg_start, seg_end, root, is_minor))
    return segments
