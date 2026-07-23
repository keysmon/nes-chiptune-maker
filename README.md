# Chiptune Maker

Turn music into NES 2A03-constrained chiptune.

The NES audio chip has exactly four usable voices: two pulse channels, one
triangle, one noise. Real 8-bit arrangements sound the way they do because a
composer chose which notes survive that budget. This tool makes those choices
automatically.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/pip install -c constraints.txt -e ".[analysis,dev]"
```

The `setuptools<81` pin in `constraints.txt` is required, not cosmetic:
`resampy` still imports `pkg_resources`, which setuptools 81 removed.

## Use

Convert a song (audio in):

```bash
.venv/bin/python -m chiptune.cli convert song.wav -o out/song_chiptune.wav
```

Or render a MIDI file directly:

```bash
.venv/bin/python -m chiptune.cli render assets/test_theme.mid -o out/theme.wav
```

`convert` separates the song into stems (Demucs), transcribes each to notes
(basic-pitch for bass/other, pyin for vocals, onset+band-energy for drums),
estimates the tempo, assembles a chip-agnostic `Score`, and renders it through
the same NES synth `render` uses. It prints the estimated BPM, per-role note
counts, and a time-resolved chroma similarity to the original.

## AI arranger (experimental, opt-in)

By default `convert` arranges the four voices with a deterministic heuristic.
An optional AI mode instead asks a language model to arrange them: it sees the
song's melody, tempo, and key, and writes a four-voice NES arrangement in
mode-relative scale-degree notation, which code parses into the same `Score` and
renders through the same synth.

```bash
.venv/bin/pip install -c constraints.txt -e ".[analysis,ai,dev]"
export GROQ_API_KEY=...        # free key from https://console.groq.com
.venv/bin/python -m chiptune.cli convert song.wav --ai -o out/song_ai.wav
```

The backend is any OpenAI-compatible endpoint, configured under `[ai]` in
`config/nes.toml` (`base_url`, `model`, `api_key_env`, `temperature`,
`max_tokens`) - Groq's free tier by default, or point it at a local Ollama
server, OpenRouter, etc. without a code change. `--ai` is shorthand for
`arrange_mode = "ai"`; setting that key in the config makes AI mode the default.

It is designed never to fail loudly: any error (missing key, network, or output
that will not parse) falls back to the heuristic arranger and logs why on stderr,
so `--ai` never crashes and never produces silence. Because the model is
non-deterministic, the arrangement is judged by ear against the heuristic - run
the same song both ways and compare. Leaving `arrange_mode = "heuristic"` (the
default) keeps the byte-for-byte deterministic path untouched.

## Web demo

Two flavors:

**Full app (local)** - upload *your own* song, convert it, and tune 26 controls live:

```bash
.venv/bin/pip install -c constraints.txt -e ".[analysis,web,dev]"
.venv/bin/python -m chiptune.web.app     # serves http://127.0.0.1:8100
```

**Hosted playground (Vercel)** - live at **https://web-demo-sandy-nine.vercel.app**.
Source in `web-demo/`. The heavy analysis (Demucs + basic-pitch: GPU, big models,
CoreML) can't run on serverless, so the hosted version ships a library of
*pre-analyzed* Creative-Commons sample songs - instrumental, vocal, and
public-domain Chinese folk (茉莉花, 送别) (see `web-demo/CREDITS.md` for the
CC-BY / CC0 / public-domain credits) - and runs only the light,
deterministic synthesis half per request (pure numpy/scipy; WAV written with the
stdlib `wave` module so there's no libsndfile dependency). Pick a sample song,
toggle the **heuristic / AI** arrangement, and tune the synthesis controls live;
converting *your own* song stays in the local app. (Vocal songs are heuristic-only
in the demo - the AI arrangements are baked offline and only for the instrumental
set so far.) Deploy from `web-demo/` with `vercel deploy --prod`.

Drop in an audio file (a 20-40 s clip works best). The server separates and
transcribes it once (~10-40 s), then you tune 26 controls across 7 groups
(harmony, arrangement, feel, vibrato, levels, drums, output) and hear the result
in ~0.25 s. Controls marked *re-analyzes* rebuild the arrangement (slower);
everything else re-synthesizes instantly from the cached `Score` - the same seam
that makes the CLI fast. This works because the slow ML analysis runs once and
the deterministic synthesis is cheap to repeat.

## Tuning the sound

Every taste-sensitive value lives in `config/nes.toml`: arpeggio rate, duty
cycles, bass octave range, quantization strength, drum voicing, drum collision
priority, channel levels. Nothing tasteful is hardcoded, so changing how it
sounds never means changing Python.

**Phantom echo** (`[echo]` in `config/nes.toml`, on by default) fills the harmony
channel's rests with a delayed copy of the lead melody, so a thin single-square-wave
lead reads fuller - the melody echoing back on itself. Each echo lands only in the
gaps between comp notes (the comp always wins the channel) and the result stays
monophonic. Turn it off with `enabled = false`, or tune `delay_frames` (how far it
trails), `volume`, and `min_lead_seconds` (skip short passing notes).

## Status

Both halves complete and merged:
- **Realization half** (`render`): MIDI/`Score` -> NES chiptune. Band-limited pulse,
  staircase triangle, LFSR noise, non-linear mixer, hardware-invariant checks.
- **Analysis half** (`convert`): audio -> `Score`. Demucs separation, basic-pitch /
  pyin / onset transcription, tempo estimation, config-gated harmony declash.
- **AI arranger** (`convert --ai`, opt-in): an LLM arranges the four voices from the
  song's melody/tempo/key over an OpenAI-compatible seam (Groq by default), with a
  guaranteed fallback to the heuristic arranger on any failure. See "AI arranger" above.

The two are joined by the serialized `Score`, so the slow ML analysis runs once and
the fast, deterministic synthesis re-renders in seconds while you tune `config/nes.toml`.

A time-resolved chroma-similarity metric (original vs chiptune) is reported as a tuning
proxy. It is not a fidelity guarantee - the real judge is listening.

Phase 1 verifies that the pipeline is *structurally* correct - the four-voice
budget is never exceeded, the triangle never varies volume, every pitch fits an
11-bit period register, the output never clips or goes NaN, and an identical
score renders byte-identically. It does not, and cannot, verify that the result
*sounds* like chiptune. That judgement is made by ear, and every knob that
affects it lives in `config/nes.toml` so tuning is a config edit, not a code
change.

## Known limitations

- **Repeated same-pitch notes fuse.** Two notes of the same pitch and role with
  no gap between them currently render as one sustained tone, because the
  per-frame pitch model cannot tell where one ends and the next begins. The fix
  is a short re-articulation gap (blanking the final frame of the earlier note),
  gated by a planned `reattack_gap_frames` value in `config/nes.toml`. The gap
  length is a taste decision best made while listening, so it is deferred to the
  tuning pass rather than shipped with an unheard default.
- **Mixer model.** The output mixer applies the NES DAC compression curve
  odd-symmetrically to the (bipolar) channel waveforms. A fully hardware-faithful
  alternative reconstructs unipolar 0-15 DAC levels and applies the exact curve;
  it is a candidate A/B for the tuning pass.

## Tests

```bash
.venv/bin/pytest -m "not slow"     # fast suite
.venv/bin/pytest                   # includes ML model loading
```
