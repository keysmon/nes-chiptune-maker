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
