"""Fast (monkeypatched) tests for Score assembly, harmony declash, and the
include_vocals=False skyline fallback. No models load here."""
from dataclasses import replace

import numpy as np
import soundfile as sf

from chiptune.analysis import build_score as bs
from chiptune.analysis.build_score import build_score, declash_harmony
from chiptune.config import load_config
from chiptune.score import NoteEvent, Percussion, Role, TempoGrid


def _tiny_song(path):
    t = np.arange(44100) / 44100
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 44100)


def _patch_common(monkeypatch):
    stems = {n: np.zeros(4410, dtype="float32") for n in ("drums", "bass", "other", "vocals")}
    monkeypatch.setattr(bs, "separate_stems", lambda p, cache_dir=None: stems)
    monkeypatch.setattr(
        bs, "transcribe_drums",
        lambda stem, sr, kb, hb, kl, hh, backtrack=True: [
            NoteEvent(38, 0.10, 0.15, 90, Role.PERCUSSION, percussion=Percussion.KICK)
        ],
    )
    monkeypatch.setattr(
        bs, "estimate_grid",
        lambda mono, sr, beats_per_bar=4: TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=beats_per_bar),
    )


def test_build_score_assembles_all_roles(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        # bass -> BASS note, other -> HARMONY note (pitch far from the lead: no clash)
        pitch = 33 if role is Role.BASS else 48
        return [NoteEvent(pitch, 0.0, 0.4, 64, role)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 0.5, 80, Role.LEAD)],
    )

    score = build_score(audio, load_config())
    roles = {n.role for n in score.notes}
    assert roles == {Role.LEAD, Role.HARMONY, Role.BASS, Role.PERCUSSION}
    assert score.tempo.bpm == 120.0
    # duration MUST be the last note end, not left at 0.0 (the allocator sizes
    # the entire output from score.duration - 0.0 truncates to a single frame).
    assert score.duration == max(n.end for n in score.notes)
    assert score.duration > 0


def test_include_vocals_false_derives_lead_from_other_skyline(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(30, 0.0, 0.4, 64, Role.BASS)]
        # `other`: two overlapping notes; the higher (72) is the skyline lead.
        return [NoteEvent(60, 0.0, 1.0, 64, Role.HARMONY), NoteEvent(72, 0.0, 1.0, 64, Role.HARMONY)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    # vocals must not be consulted when include_vocals is False
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vocals must not be called")),
    )

    cfg = load_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, include_vocals=False))
    score = build_score(audio, cfg)

    lead = [n for n in score.notes if n.role is Role.LEAD]
    assert lead, "LEAD must be derived from `other` when include_vocals is False"
    assert max(n.pitch for n in lead) == 72  # the top voice
    assert any(n.role is Role.HARMONY for n in score.notes)  # lower voice stays HARMONY


def _n(pitch, start, end, role):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=64, role=role)


def test_declash_pushes_only_clashing_harmony_down_one_octave():
    lead = _n(60, 0.0, 1.0, Role.LEAD)
    clash = _n(61, 0.5, 1.5, Role.HARMONY)      # overlaps + within 1 semitone -> -12
    no_overlap = _n(61, 2.0, 3.0, Role.HARMONY)  # within pitch but no time overlap -> keep
    far = _n(72, 0.0, 1.0, Role.HARMONY)         # overlaps but >1 semitone away -> keep

    out = declash_harmony([lead, clash, no_overlap, far], declash_semitones=1)
    by = {(n.role, round(n.start, 3)): n.pitch for n in out}
    assert by[(Role.HARMONY, 0.5)] == 49   # 61 - 12
    assert by[(Role.HARMONY, 2.0)] == 61   # untouched (no time overlap)
    assert by[(Role.HARMONY, 0.0)] == 72   # untouched (out of semitone range)
    assert by[(Role.LEAD, 0.0)] == 60      # leads are never modified


def test_declash_guards_octave_underflow():
    # harmony within range but pitch < 12: pushing down would underflow MIDI 0,
    # so it must stay put rather than raise.
    out = declash_harmony([_n(6, 0.0, 1.0, Role.LEAD), _n(6, 0.0, 1.0, Role.HARMONY)], 1)
    harm = [n for n in out if n.role is Role.HARMONY][0]
    assert harm.pitch == 6
