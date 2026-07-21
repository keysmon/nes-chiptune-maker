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

```bash
.venv/bin/python -m chiptune.cli render assets/test_theme.mid -o out/theme.wav
```

## Tuning the sound

Every taste-sensitive value lives in `config/nes.toml`: arpeggio rate, duty
cycles, bass octave range, quantization strength, drum voicing, drum collision
priority, channel levels. Nothing tasteful is hardcoded, so changing how it
sounds never means changing Python.

## Status

Phase 1 complete: MIDI in, chiptune out. Source separation and audio
transcription (audio in) are the next plan.

## Tests

```bash
.venv/bin/pytest -m "not slow"     # fast suite
.venv/bin/pytest                   # includes ML model loading
```
