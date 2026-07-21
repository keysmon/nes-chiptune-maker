import pytest

from chiptune.config import load_config
from chiptune.nes.tables import playable_on_pulse
from chiptune.score import NoteEvent, Role, Score, TempoGrid
from chiptune.arrange.timeline import ChannelId, FrameEvent
from chiptune.arrange.allocator import allocate, fold_into_range


def score_of(notes, duration=2.0):
    return Score(tempo=TempoGrid(120.0, 0.0, 4), notes=notes, duration=duration)


@pytest.fixture
def cfg():
    return load_config()


def test_fold_into_range_raises_low_notes_by_octaves():
    assert fold_into_range(12, low=28, high=55) == 36     # +2 octaves
    assert fold_into_range(24, low=28, high=55) == 36


def test_fold_into_range_lowers_high_notes_by_octaves():
    assert fold_into_range(72, low=28, high=55) == 48


def test_fold_into_range_leaves_in_range_notes_alone():
    assert fold_into_range(40, low=28, high=55) == 40


def test_lead_note_occupies_its_frames_on_pulse1(cfg):
    s = score_of([NoteEvent(72, 0.0, 0.5, 100, Role.LEAD)])
    tl = allocate(s, cfg)[ChannelId.PULSE1]
    # 60 fps, 0.5 s -> frames 0..29 sounding, frame 30 silent
    assert tl.frames[0].pitch == 72
    assert tl.frames[29].pitch == 72
    assert tl.frames[30].pitch is None


def test_overlapping_lead_notes_keep_the_highest(cfg):
    s = score_of([
        NoteEvent(72, 0.0, 0.5, 100, Role.LEAD),
        NoteEvent(76, 0.0, 0.5, 100, Role.LEAD),
        NoteEvent(69, 0.0, 0.5, 100, Role.LEAD),
    ])
    tl = allocate(s, cfg)[ChannelId.PULSE1]
    assert tl.frames[5].pitch == 76


def test_overlapping_bass_notes_keep_the_lowest(cfg):
    s = score_of([
        NoteEvent(36, 0.0, 0.5, 100, Role.BASS),
        NoteEvent(48, 0.0, 0.5, 100, Role.BASS),
    ])
    tl = allocate(s, cfg)[ChannelId.TRIANGLE]
    assert tl.frames[5].pitch == 36


def test_bass_is_folded_into_the_triangle_range(cfg):
    s = score_of([NoteEvent(12, 0.0, 0.5, 100, Role.BASS)])
    tl = allocate(s, cfg)[ChannelId.TRIANGLE]
    assert cfg.arrange.bass_low <= tl.frames[0].pitch <= cfg.arrange.bass_high


def test_triangle_volume_never_varies(cfg):
    """Hardware invariant: the triangle channel has no volume control."""
    s = score_of([
        NoteEvent(36, 0.0, 0.5, 20, Role.BASS),
        NoteEvent(38, 0.5, 1.0, 127, Role.BASS),
    ])
    tl = allocate(s, cfg)[ChannelId.TRIANGLE]
    volumes = {f.volume for f in tl.frames if f.pitch is not None}
    assert len(volumes) == 1, f"triangle volume varied: {volumes}"


def test_six_simultaneous_lead_notes_reduce_to_one_sounding_pitch(cfg):
    """Hardware invariant, spec 6.1: one note per channel per frame."""
    s = score_of([NoteEvent(60 + i, 0.0, 0.5, 100, Role.LEAD) for i in range(6)])
    tl = allocate(s, cfg)[ChannelId.PULSE1]
    sounding = [f.pitch for f in tl.frames[:30]]
    assert sounding == [65] * 30, "must reduce to the single highest pitch (60+5)"
    assert all(f.pitch is None for f in tl.frames[30:])


def test_all_four_channels_are_always_returned(cfg):
    s = score_of([NoteEvent(72, 0.0, 0.5, 100, Role.LEAD)])
    out = allocate(s, cfg)
    assert set(out) == set(ChannelId)


def test_unplayable_pitches_are_dropped_not_clamped(cfg):
    """A pitch outside the register range must not silently become a wrong note."""
    s = score_of([NoteEvent(126, 0.0, 0.5, 100, Role.BASS)])
    tl = allocate(s, cfg)[ChannelId.TRIANGLE]
    sounded = [f.pitch for f in tl.frames if f.pitch is not None]
    assert all(cfg.arrange.bass_low <= p <= cfg.arrange.bass_high for p in sounded)


def test_unplayable_lead_pitch_is_dropped_to_silence(cfg):
    """Exercises the only reachable drop path.

    Bass pitches are octave-folded into a range that is always playable, so
    that drop branch can't fire; lead notes are filtered without folding, so
    an unplayable lead pitch must silence its frames rather than sound a
    wrong (clamped) note.

    MIDI pitch 20 is far too low for the pulse channel: its exact period is
    ~4308, well above the 11-bit register max of 2047.
    """
    assert not playable_on_pulse(20), "pitch 20 must be genuinely unplayable on pulse"
    s = score_of([NoteEvent(20, 0.0, 0.5, 100, Role.LEAD)])
    tl = allocate(s, cfg)[ChannelId.PULSE1]
    assert all(f.pitch is None for f in tl.frames)


def test_velocity_scales_pulse_volume(cfg):
    soft = Score(TempoGrid(120., 0., 4), [NoteEvent(72, 0., 0.5, 30, Role.LEAD)], 1.0)
    loud = Score(TempoGrid(120., 0., 4), [NoteEvent(72, 0., 0.5, 127, Role.LEAD)], 1.0)
    vs = allocate(soft, cfg)[ChannelId.PULSE1].frames[5].volume
    vl = allocate(loud, cfg)[ChannelId.PULSE1].frames[5].volume
    assert vs < vl, "louder velocity must give higher volume"
    assert vs > 0, "velocity_floor keeps soft notes audible"


def test_triangle_ignores_velocity(cfg):
    s = Score(TempoGrid(120., 0., 4),
              [NoteEvent(36, 0., 0.5, 20, Role.BASS), NoteEvent(38, 0.5, 1.0, 127, Role.BASS)], 1.0)
    vols = {f.volume for f in allocate(s, cfg)[ChannelId.TRIANGLE].frames if f.pitch is not None}
    assert len(vols) == 1, "triangle volume must not vary with velocity (no hardware volume)"


def test_reattack_gap_creates_a_silent_frame_between_repeated_lead_notes(cfg):
    """Wiring check: allocate() must apply rearticulate() before reducing to frames."""
    s = score_of([
        NoteEvent(72, 0.0, 0.5, 100, Role.LEAD),
        NoteEvent(72, 0.5, 1.0, 100, Role.LEAD),
    ])
    tl = allocate(s, cfg)[ChannelId.PULSE1]
    assert tl.frames[28].pitch == 72
    assert tl.frames[29].pitch is None, "a re-attack gap must silence a frame before the repeat"
    assert tl.frames[30].pitch == 72


def test_reattack_gap_applies_to_bass_too(cfg):
    s = score_of([
        NoteEvent(36, 0.0, 0.5, 100, Role.BASS),
        NoteEvent(36, 0.5, 1.0, 100, Role.BASS),
    ])
    tl = allocate(s, cfg)[ChannelId.TRIANGLE]
    assert tl.frames[28].pitch == 36
    assert tl.frames[29].pitch is None, "bass re-attack must also get a silent gap frame"
    assert tl.frames[30].pitch == 36


def test_dropped_unplayable_note_warns_to_stderr(cfg, capsys):
    """Dropping is correct, but - like midi_io's percussion warning - not silent.

    An out-of-range pitch must be reported so a Phase-2 transcription feeding real
    pitches does not silently lose notes. Mirrors the stderr warning style used
    when midi_io drops unmapped percussion.
    """
    assert not playable_on_pulse(20), "pitch 20 must be genuinely unplayable on pulse"
    s = score_of([NoteEvent(20, 0.0, 0.5, 100, Role.LEAD)])
    allocate(s, cfg)
    err = capsys.readouterr().err
    assert "20" in err, f"warning must name the dropped pitch; got: {err!r}"
    assert "lead" in err.lower(), f"warning must name the role; got: {err!r}"
