"""Command line entry point.

No os.system anywhere: every stage is a Python call, so failures surface as
exceptions with stack traces rather than silence. This is a direct response to
the 2022 project, whose two broken stages shelled out and never checked exit
codes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .arrange.allocator import allocate
from .arrange.timeline import ChannelId
from .config import load_config
from .invariants import check_invariants
from .midi_io import load_midi
from .quantize import quantize_score
from .score import Score
from .synth.apu import render_channels
from .synth.mixer import nes_mix, write_wav


def render_score(score: Score, config_path=None, out_path="out/chiptune.wav") -> Path:
    cfg = load_config(config_path)

    quantized = quantize_score(
        score,
        subdivision=cfg.arrange.subdivision,
        strength=cfg.arrange.quantize_strength,
        min_duration=cfg.arrange.min_duration,
    )
    timelines = allocate(quantized, cfg)
    channels = render_channels(timelines, cfg)
    mixed = nes_mix(
        channels[ChannelId.PULSE1],
        channels[ChannelId.PULSE2],
        channels[ChannelId.TRIANGLE],
        channels[ChannelId.NOISE],
    )
    check_invariants(timelines, mixed)

    out = Path(out_path)
    write_wav(out, mixed, cfg.sample_rate)
    return out


def render_midi(midi_path, config_path=None, out_path="out/chiptune.wav") -> Path:
    return render_score(load_midi(midi_path), config_path, out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chiptune", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a MIDI file to NES chiptune")
    render.add_argument("input", help="path to a .mid file")
    render.add_argument("-o", "--output", default="out/chiptune.wav")
    render.add_argument("-c", "--config", default=None, help="path to a nes.toml")

    args = parser.parse_args(argv)

    if args.command == "render":
        try:
            score = load_midi(args.input)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        out = render_score(score, args.config, args.output)
        print(f"tempo: {score.tempo.bpm:.1f} BPM   notes: {len(score.notes)}")
        print(f"wrote {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
