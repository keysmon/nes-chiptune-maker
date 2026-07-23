"""Fast (monkeypatched) tests for Score assembly, harmony declash, and the
include_vocals=False skyline fallback. No models load here."""
from dataclasses import replace

import numpy as np
import soundfile as sf

from chiptune.analysis import build_score as bs
from chiptune.analysis.build_score import build_score, declash_harmony, smooth_lead_leaps
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


def test_smooth_lead_leaps_folds_octave_jumps():
    # A >octave artifact jump (60 -> 84) is folded toward the previous note; sub-octave
    # leaps and pitch class are preserved, and max_leap=0 is a no-op.
    lead = [NoteEvent(60, 0., .5, 100, Role.LEAD),
            NoteEvent(84, .5, 1., 100, Role.LEAD),   # 2 octaves up = skyline artifact
            NoteEvent(62, 1., 1.5, 100, Role.LEAD)]
    assert [n.pitch for n in smooth_lead_leaps(lead, 12)] == [60, 72, 62]  # 84 -> 72 (within an octave)
    assert [n.pitch for n in smooth_lead_leaps(lead, 0)] == [60, 84, 62]   # off = unchanged

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

    # harmony_mode="transcribe": this test is about role assembly from the
    # basic-pitch stems, not the chord path (covered separately below).
    cfg = load_config()
    cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_mode="transcribe"))
    score = build_score(audio, cfg)
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

    # harmony_mode="transcribe": exercises the skyline-split mechanics
    # (lower voice of `other` staying HARMONY), which only the transcribe
    # path does - see test_chords_mode_discards_skyline_harmony_half below
    # for the equivalent chords-mode behavior.
    cfg = load_config()
    cfg = replace(
        cfg,
        analysis=replace(cfg.analysis, include_vocals=False),
        arrange=replace(cfg.arrange, harmony_mode="transcribe"),
    )
    score = build_score(audio, cfg)

    lead = [n for n in score.notes if n.role is Role.LEAD]
    assert lead, "LEAD must be derived from `other` when include_vocals is False"
    assert max(n.pitch for n in lead) == 72  # the top voice
    assert any(n.role is Role.HARMONY for n in score.notes)  # lower voice stays HARMONY


def _fake_wall_of_harmony(n=50):
    """The "muddy HARMONY" problem: a basic-pitch transcription of `other`
    would return dozens of short, overlapping notes."""
    return [NoteEvent(48 + i % 12, i * 0.02, i * 0.02 + 0.05, 60, Role.HARMONY) for i in range(n)]


def test_harmony_mode_chords_yields_far_fewer_notes_than_transcribe(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]
        return _fake_wall_of_harmony()

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 1.0, 80, Role.LEAD)],
    )

    cfg = load_config()
    chords_cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_mode="chords"))
    transcribe_cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_mode="transcribe"))

    # detect_chords/comp_chords run for real here (same as tests/test_chords.py
    # in this same fast suite) - cheap librosa chroma, no ML model.
    chords_score = build_score(audio, chords_cfg)
    transcribe_score = build_score(audio, transcribe_cfg)

    chords_harmony = chords_score.notes_with_role(Role.HARMONY)
    transcribe_harmony = transcribe_score.notes_with_role(Role.HARMONY)

    assert len(transcribe_harmony) == 50  # sanity: the raw wall passed through unmodified
    assert 0 < len(chords_harmony) < len(transcribe_harmony)

    from chiptune.analysis.density import score_density
    chords_density = score_density(chords_score, frame_rate=cfg.frame_rate)
    transcribe_density = score_density(transcribe_score, frame_rate=cfg.frame_rate)
    assert chords_density["mean_simultaneous"] < transcribe_density["mean_simultaneous"]


def test_chords_mode_arp_source_never_transcribes_other_when_include_vocals(monkeypatch, tmp_path):
    """harmony_source='arp' (the legacy fixed-arpeggio fallback) builds HARMONY
    purely from detected chords and must never basic-pitch transcribe the
    'other' stem. harmony_source='select' (the default) deliberately DOES
    transcribe 'other' for HARMONY candidates on the vocal path - see
    test_chords_mode_select_source_transcribes_other_when_include_vocals below."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.HARMONY:
            raise AssertionError("harmony_source='arp' must not basic-pitch transcribe 'other'")
        return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 1.0, 80, Role.LEAD)],
    )

    cfg = load_config()  # default harmony_mode is "chords"
    cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_source="arp"))
    score = build_score(audio, cfg)

    assert score.notes_with_role(Role.HARMONY)  # still gets a comped harmony


def test_chords_mode_select_source_transcribes_other_when_include_vocals(monkeypatch, tmp_path):
    """harmony_source='select' (the default) has no skyline_harmony to reuse on
    the vocal path (there's no 'other' skyline split when vocals carry LEAD), so
    it must basic-pitch transcribe 'other' for Role.HARMONY candidates - the
    inverse of the 'arp' guarantee above."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    calls = []

    def fake_pitched(stem, sr, role, min_duration=0.0):
        calls.append(role)
        if role is Role.HARMONY:
            return [NoteEvent(60, 0.0, 0.4, 100, Role.HARMONY)]
        return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 1.0, 80, Role.LEAD)],
    )

    cfg = load_config()  # default harmony_mode="chords", harmony_source="select"
    build_score(audio, cfg)

    assert Role.HARMONY in calls  # 'other' WAS transcribed for HARMONY candidates


def test_chords_mode_discards_skyline_harmony_half(monkeypatch, tmp_path):
    """include_vocals=False + harmony_mode='chords' + harmony_source='arp':
    `other` is still transcribed to derive the skyline LEAD (there is no
    other melody source on the instrumental path), but its covered/lower
    half (skyline_harmony) must NOT populate HARMONY - under 'arp' HARMONY
    comes entirely from the chord-comp arpeggio (`comp_chords`), which
    structurally never reads skyline_harmony/other_candidates at all. This
    pins the same invariant as
    test_chords_mode_arp_source_never_transcribes_other_when_include_vocals
    above, but on the include_vocals=False path where skyline_harmony
    (rather than a full basic-pitch 'other' transcription) is the material
    that must be discarded.

    NOTE: under harmony_source='select' (the default) this invariant does
    NOT hold - select_comp reuses skyline_harmony as its candidate pool and
    snaps it into HARMONY; see
    test_chords_mode_select_source_transcribes_other_when_include_vocals."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(30, 0.0, 0.4, 64, Role.BASS)]
        return [NoteEvent(60, 0.0, 1.0, 64, Role.HARMONY), NoteEvent(72, 0.0, 1.0, 64, Role.HARMONY)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vocals must not be called")),
    )

    cfg = load_config()
    cfg = replace(
        cfg,
        analysis=replace(cfg.analysis, include_vocals=False),
        arrange=replace(cfg.arrange, harmony_source="arp"),
    )
    score = build_score(audio, cfg)

    lead = [n for n in score.notes if n.role is Role.LEAD]
    assert lead and max(n.pitch for n in lead) == 72  # skyline lead still derived

    harmony = score.notes_with_role(Role.HARMONY)
    assert harmony, "must still get a comped harmony"
    # comp_chords hardcodes velocity=_HARMONY_VELOCITY (80) for every note it
    # emits; the fixture's raw `other` transcription above uses velocity=64.
    # A velocity-64 note in HARMONY would prove the discarded skyline harmony
    # half leaked through instead of being replaced by the chord comp. This
    # holds regardless of what the detected chord's pitch classes happen to
    # be - unlike checking `pitch != 60`, which only passed before by
    # coincidence of the sine's detected chord not containing pitch class C.
    assert all(n.velocity == 80 for n in harmony), (
        "HARMONY must come entirely from the chord-comp arpeggio (velocity=80), "
        "not the discarded skyline harmony half (velocity=64)"
    )


def test_build_score_applies_melody_thinning_and_bass_simplification(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            # a 10ms glitch note between two longer same-pitch notes
            return [
                NoteEvent(30, 0.0, 0.5, 64, Role.BASS),
                NoteEvent(35, 0.5, 0.51, 64, Role.BASS),
                NoteEvent(30, 0.52, 1.0, 64, Role.BASS),
            ]
        return _fake_wall_of_harmony()

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [
            NoteEvent(67, 0.0, 0.5, 80, Role.LEAD),
            NoteEvent(72, 0.5, 0.51, 80, Role.LEAD),  # 10ms ornament
            NoteEvent(69, 0.52, 1.0, 80, Role.LEAD),
        ],
    )

    cfg = load_config()  # melody_min_seconds/bass_min_seconds default to 0.08
    score = build_score(audio, cfg)

    lead = score.notes_with_role(Role.LEAD)
    bass = score.notes_with_role(Role.BASS)
    assert 72 not in [n.pitch for n in lead], "10ms LEAD ornament must be thinned out"
    assert 35 not in [n.pitch for n in bass], "10ms BASS glitch must be thinned out"
    assert len(bass) == 1 and bass[0].start == 0.0 and bass[0].end == 1.0  # merged


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


def test_ai_mode_uses_llm_arrangement(monkeypatch, tmp_path):
    """arrange_mode='ai': the heuristic Score is still built first (it's the
    fallback + melody/tempo source), then handed to ai_arranger.arrange, whose
    (mocked) LLM output replaces it wholesale."""
    from chiptune.arrange import ai_arranger

    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        pitch = 33 if role is Role.BASS else 48
        return [NoteEvent(pitch, 0.0, 0.4, 64, role)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 0.5, 80, Role.LEAD)],
    )
    monkeypatch.setattr(
        ai_arranger, "_call_llm",
        lambda prompt, cfg: "KEY: C maj\nLEAD: 1:1 3:1\nBASS: 1:2\nDRUMS: K:1 S:1",
    )

    cfg = load_config()
    cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_mode="transcribe", arrange_mode="ai"))
    score = build_score(audio, cfg)

    assert any(n.role is Role.BASS for n in score.notes)
    # the AI-parsed arrangement replaces the heuristic notes wholesale: the
    # heuristic vocal pitch (67) must not survive, and the LLM's own LEAD
    # degree-1-in-C (pitch 60) must be present instead.
    assert 67 not in [n.pitch for n in score.notes_with_role(Role.LEAD)]
    assert 60 in [n.pitch for n in score.notes_with_role(Role.LEAD)]


def test_heuristic_mode_unchanged(monkeypatch, tmp_path):
    """arrange_mode='heuristic' (the default): identical to pre-AI behavior -
    no ai_arranger call, no LLM output substitution."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        pitch = 33 if role is Role.BASS else 48
        return [NoteEvent(pitch, 0.0, 0.4, 64, role)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 0.5, 80, Role.LEAD)],
    )

    cfg = load_config()  # arrange_mode defaults to "heuristic"
    cfg = replace(cfg, arrange=replace(cfg.arrange, harmony_mode="transcribe"))
    score = build_score(audio, cfg)

    assert 67 in [n.pitch for n in score.notes_with_role(Role.LEAD)]  # untouched by AI


def test_phantom_echo_adds_harmony_only_in_gaps_when_enabled(monkeypatch, tmp_path):
    """[echo].enabled=true adds delayed lead echoes to HARMONY (filling the comp's
    gaps) and leaves LEAD/BASS/PERCUSSION untouched; the HARMONY list stays
    monophonic and the original comp note is preserved."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]
        return [NoteEvent(48, 0.0, 0.4, 64, Role.HARMONY)]  # one comp note, gap after 0.4

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 0.5, 80, Role.LEAD)],
    )

    # transcribe mode = predictable HARMONY (the raw fake note); declash off so
    # the echo pitch is not octave-shifted, keeping the assertions simple.
    cfg = load_config()
    cfg = replace(
        cfg,
        arrange=replace(cfg.arrange, harmony_mode="transcribe"),
        analysis=replace(cfg.analysis, harmony_declash=False),
    )
    off = build_score(audio, replace(cfg, echo=replace(cfg.echo, enabled=False)))
    on = build_score(audio, replace(cfg, echo=replace(cfg.echo, enabled=True)))

    for role in (Role.LEAD, Role.BASS, Role.PERCUSSION):
        assert on.notes_with_role(role) == off.notes_with_role(role), f"{role} must be untouched"

    off_harm = off.notes_with_role(Role.HARMONY)
    on_harm = sorted(on.notes_with_role(Role.HARMONY), key=lambda n: n.start)
    assert len(on_harm) > len(off_harm), "echo must add at least one HARMONY note"
    assert off_harm[0] in on_harm, "comp note preserved (comp wins)"
    for a, b in zip(on_harm, on_harm[1:]):
        assert a.end <= b.start, "HARMONY must stay strictly monophonic"


def test_phantom_echo_disabled_leaves_harmony_identical_to_the_comp(monkeypatch, tmp_path):
    """The default (disabled) path is a no-op: HARMONY equals the plain comp."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]
        return [NoteEvent(48, 0.0, 0.4, 64, Role.HARMONY)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 0.5, 80, Role.LEAD)],
    )

    cfg = load_config()  # echo disabled by default
    cfg = replace(
        cfg,
        arrange=replace(cfg.arrange, harmony_mode="transcribe"),
        analysis=replace(cfg.analysis, harmony_declash=False),
    )
    score = build_score(audio, cfg)
    harm = score.notes_with_role(Role.HARMONY)
    assert harm == [NoteEvent(48, 0.0, 0.4, 64, Role.HARMONY)]  # untouched comp only


def test_phantom_echo_survives_rest_on_busy_melody(monkeypatch, tmp_path):
    """Echo runs AFTER rest_harmony_on_busy_melody and is exempt from it: with a
    lead spanning the song and rest-on-busy stripping the comp, the delayed
    echoes still populate HARMONY (the echo re-states the melody by design)."""
    audio = tmp_path / "song.wav"
    _tiny_song(audio)
    _patch_common(monkeypatch)

    def fake_pitched(stem, sr, role, min_duration=0.0):
        if role is Role.BASS:
            return [NoteEvent(33, 0.0, 0.4, 64, Role.BASS)]
        return [NoteEvent(48, 0.0, 0.4, 64, Role.HARMONY)]

    monkeypatch.setattr(bs, "transcribe_pitched", fake_pitched)
    monkeypatch.setattr(
        bs, "transcribe_vocals",
        lambda stem, sr, fmin, fmax, min_duration=0.06: [NoteEvent(67, 0.0, 1.0, 80, Role.LEAD)],
    )

    cfg = load_config()
    cfg = replace(
        cfg,
        arrange=replace(cfg.arrange, harmony_mode="transcribe",
                        harmony_rest_on_busy_melody=True),
        analysis=replace(cfg.analysis, harmony_declash=False),
        echo=replace(cfg.echo, enabled=True),
    )
    score = build_score(audio, cfg)
    harm = sorted(score.notes_with_role(Role.HARMONY), key=lambda n: n.start)
    assert any(n.pitch == 67 for n in harm), "the delayed lead echo must survive the rest pass"
    for a, b in zip(harm, harm[1:]):
        assert a.end <= b.start
