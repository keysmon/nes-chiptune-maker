"""Chiptune Maker - hosted playground (Vercel ASGI entrypoint).

The heavy analysis half (Demucs + basic-pitch: GPU, big models, CoreML) can't run
on serverless, so this demo ships PRE-ANALYZED Scores (offline, per song, in both
harmony modes) and runs only the light deterministic synthesis half - pure
numpy/scipy - per request. Synthesis knobs re-render instantly; the harmony toggle
swaps the two pre-baked Scores. Live song upload runs in the local full app.
"""
import io
import os
import re
import tomllib
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from chiptune.arrange.allocator import allocate
from chiptune.arrange.timeline import ChannelId
from chiptune.config import _deep_merge, config_from_dict
from chiptune.invariants import check_invariants
from chiptune.quantize import quantize_score
from chiptune.score import Score
from chiptune.synth.apu import render_channels
from chiptune.synth.mixer import apply_output_filter, nes_mix

_BASE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_BASE, "static")
_DEFAULT = tomllib.load(open(os.path.join(_BASE, "nes.toml"), "rb"))
_SCORES: dict[str, Score] = {}
_VALID_HARMONY = ("chords", "transcribe", "ai")
_SLUG = re.compile(r"^[a-z0-9]+$")

app = FastAPI(title="Chiptune Maker playground")


def _load_score(song: str, harmony: str) -> Score:
    key = f"{song}-{harmony}"
    if key not in _SCORES:
        path = os.path.join(_BASE, "scores", f"{key}.json")
        if not os.path.exists(path):
            raise HTTPException(400, f"unknown song/harmony: {key}")
        _SCORES[key] = Score.from_json(open(path).read())
    return _SCORES[key]


def _render(score: Score, overrides: dict) -> bytes:
    try:
        cfg = config_from_dict(_deep_merge(_DEFAULT, overrides or {}))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
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
    return _wav_bytes(mix, cfg.sample_rate)


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Write a mono 16-bit PCM WAV with the stdlib `wave` module - no soundfile /
    libsndfile, which isn't available on the serverless runtime."""
    ints = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())
    return buf.getvalue()


class RenderRequest(BaseModel):
    song: str = "pop"
    harmony: str = "chords"
    overrides: dict = {}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/compare")
def compare():
    return FileResponse(os.path.join(_STATIC, "compare.html"))


@app.get("/original/{song}")
def original(song: str):
    if not _SLUG.match(song):
        raise HTTPException(400, "bad song id")
    path = os.path.join(_STATIC, f"{song}-original.mp3")
    if not os.path.exists(path):
        raise HTTPException(404, "no such song")
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/render")
def render(req: RenderRequest):
    if req.harmony not in _VALID_HARMONY:
        raise HTTPException(400, f"harmony must be one of {_VALID_HARMONY}")
    score = _load_score(req.song, req.harmony)
    wav = _render(score, req.overrides or {})
    return Response(content=wav, media_type="audio/wav")
