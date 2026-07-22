"""Web runtime: turn an uploaded song + a config-override dict into chiptune bytes.

Leverages the Score seam. On upload we run the slow analysis (Demucs + transcription
+ chord detection) once and cache the resulting Score. A config change that only
affects SYNTHESIS (levels, vibrato, drum voicing, output filter, quantize, ...)
re-renders from the cached Score in ~1s; a change that affects ANALYSIS (harmony
mode, chord params, thinning, declash, vocals) rebuilds the Score (~15s). The
`analysis signature` decides which path a request takes.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..analysis.build_score import build_score
from ..analysis.density import score_density
from ..arrange.allocator import allocate
from ..arrange.timeline import ChannelId
from ..config import Config, config_from_overrides
from ..invariants import check_invariants
from ..quantize import quantize_score
from ..score import Score
from ..synth.apu import render_channels
from ..synth.mixer import apply_output_filter, nes_mix

# [arrange] keys that change the SCORE (analysis half) - a change to any of these
# rebuilds the Score. Every OTHER [arrange] key is synthesis-only (fast re-render).
_ANALYSIS_ARRANGE_KEYS = frozenset({
    "harmony_mode", "chord_comp_pattern", "chord_subdivision", "chord_octave",
    "chord_tones", "chord_smooth_beats", "melody_min_seconds", "bass_min_seconds",
    "harmony_rest_on_busy_melody", "arrange_mode", "lead_max_leap",
})
# [arrange] keys consumed only by quantize/allocate/synth (a change re-renders from
# the cached Score). Kept explicit + partition-tested against ArrangeConfig so that
# adding a new [arrange] field forces a conscious analysis-vs-synthesis choice
# rather than silently defaulting to "synthesis" and skipping a needed re-analyze.
_SYNTHESIS_ARRANGE_KEYS = frozenset({
    "subdivision", "quantize_strength", "min_duration", "arpeggio_frames",
    "bass_low", "bass_high", "borrow_enabled", "borrow_idle_frames",
    "borrow_hysteresis_frames", "velocity_floor", "reattack_gap",
})

_MAX_SESSIONS = 32  # cap in-memory sessions + temp files (LRU-evict the oldest)


def _analysis_signature(cfg: Config) -> str:
    arr = {k: getattr(cfg.arrange, k) for k in sorted(_ANALYSIS_ARRANGE_KEYS)}
    ana = {
        k: getattr(cfg.analysis, k)
        for k in sorted(vars(cfg.analysis)) if not k.startswith("_")
    }
    blob = json.dumps({"arrange": arr, "analysis": ana}, default=str, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def render_wav_bytes(score: Score, cfg: Config) -> bytes:
    """Run the (fast, deterministic) realization half and return WAV bytes."""
    quantized = quantize_score(
        score,
        subdivision=cfg.arrange.subdivision,
        strength=cfg.arrange.quantize_strength,
        min_duration=cfg.arrange.min_duration,
    )
    timelines = allocate(quantized, cfg)
    channels = render_channels(timelines, cfg)
    mix = nes_mix(
        channels[ChannelId.PULSE1], channels[ChannelId.PULSE2],
        channels[ChannelId.TRIANGLE], channels[ChannelId.NOISE],
    )
    mix = apply_output_filter(mix, cfg.sample_rate,
                              highpass_hz=cfg.output_highpass_hz,
                              lowpass_hz=cfg.output_lowpass_hz)
    check_invariants(timelines, mix)
    wav = np.clip(mix, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), cfg.sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@dataclass
class Session:
    session_id: str
    audio_path: Path
    duration: float
    score: Score | None = None
    analysis_sig: str | None = None


class SessionStore:
    """In-memory session registry. Each session owns an uploaded file, a stem cache
    dir, and the most recent Score (rebuilt only when analysis params change)."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.work_dir / "stems"
        self._sessions: dict[str, Session] = {}

    def create(self, filename: str, data: bytes) -> Session:
        sid = hashlib.sha256(data + filename.encode()).hexdigest()[:16]
        ext = Path(filename).suffix.lower() or ".wav"
        audio_path = self.work_dir / f"{sid}{ext}"
        audio_path.write_bytes(data)
        try:
            y, sr = sf.read(str(audio_path))
        except sf.SoundFileError:
            audio_path.unlink(missing_ok=True)
            raise  # app layer maps to 400
        dur = len(y) / sr
        sess = Session(session_id=sid, audio_path=audio_path, duration=round(dur, 2))
        self._sessions[sid] = sess
        self._evict_if_needed()
        return sess

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > _MAX_SESSIONS:
            old_sid, old = next(iter(self._sessions.items()))
            self._sessions.pop(old_sid, None)
            old.audio_path.unlink(missing_ok=True)

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def render(self, sid: str, overrides: dict) -> tuple[bytes, dict]:
        """Build a Config from overrides, (re)analyze only if the analysis signature
        changed, then render. Returns (wav_bytes, stats)."""
        sess = self._sessions.get(sid)
        if sess is None:
            raise KeyError(sid)
        cfg = config_from_overrides(overrides)  # raises ValueError on a bad value
        sig = _analysis_signature(cfg)
        t0 = time.monotonic()
        reanalyzed = False
        if sess.score is None or sess.analysis_sig != sig:
            sess.score = build_score(sess.audio_path, cfg, cache_dir=self.cache_dir)
            sess.analysis_sig = sig
            reanalyzed = True
        analyze_s = time.monotonic() - t0
        t1 = time.monotonic()
        wav = render_wav_bytes(sess.score, cfg)
        render_s = time.monotonic() - t1

        dens = score_density(sess.score)
        by_role: dict[str, int] = {}
        for n in sess.score.notes:
            by_role[n.role.value] = by_role.get(n.role.value, 0) + 1
        stats = {
            "reanalyzed": reanalyzed,
            "analyze_seconds": round(analyze_s, 2),
            "render_seconds": round(render_s, 2),
            "bpm": round(sess.score.tempo.bpm, 1),
            "notes_by_role": by_role,
            "mean_simultaneous": round(dens["mean_simultaneous"], 2),
        }
        return wav, stats
