"""FastAPI app for the Chiptune Maker web demo.

Flow: upload a song -> the server separates + transcribes it once -> you move
config sliders -> the server re-synthesizes (fast) or re-analyzes (when a harmony/
arrangement knob changes) and streams back the chiptune WAV plus stats.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .runtime import SessionStore
from .schema import schema_with_defaults

STATIC_DIR = Path(__file__).parent / "static"
WORK_DIR = Path(tempfile.gettempdir()) / "chiptune_web"

app = FastAPI(title="Chiptune Maker")
store = SessionStore(WORK_DIR)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def expand_paths(flat: dict) -> dict:
    """Turn a flat {"arrange.chord_octave": 4, "drums.snare.volume": 9} map into a
    nested override dict."""
    nested: dict = {}
    for path, value in flat.items():
        parts = path.split(".")
        node = nested
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return nested


class RenderRequest(BaseModel):
    session_id: str
    overrides: dict = {}  # flat {dot.path: value}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/schema")
def get_schema():
    return {"controls": schema_with_defaults()}


@app.post("/api/upload")
async def upload(file: UploadFile):
    # Read in bounded chunks and stop at the cap, so an oversized POST can't spool
    # gigabytes into memory before we reject it.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        sess = store.create(file.filename or "upload.wav", data)
    except sf.SoundFileError as exc:  # not decodable audio
        raise HTTPException(400, f"could not read audio: {exc}")
    return {
        "session_id": sess.session_id,
        "duration": sess.duration,
        "original_url": f"/api/original/{sess.session_id}",
    }


@app.get("/api/original/{sid}")
def original(sid: str):
    sess = store.get(sid)
    if sess is None:
        raise HTTPException(404, "unknown session")
    return FileResponse(sess.audio_path)


@app.post("/api/render")
def render(req: RenderRequest):
    try:
        wav, stats = store.render(req.session_id, expand_paths(req.overrides))
    except KeyError:
        raise HTTPException(404, "unknown session")
    except (ValueError, TypeError) as exc:  # invalid / unknown config value
        raise HTTPException(400, str(exc))
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-Chiptune-Stats": json.dumps(stats)},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
