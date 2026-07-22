from chiptune.analysis.chords import ChordSegment
from chiptune.arrange.comp_select import _tone_pcs, _chord_at, _snap

C_MAJ = ChordSegment(start=0.0, end=1.0, root=0, is_minor=False)   # C E G -> pcs {0,4,7}
A_MIN = ChordSegment(start=1.0, end=2.0, root=9, is_minor=True)    # A C E -> pcs {9,0,4}


def test_tone_pcs_major_and_minor():
    assert _tone_pcs(C_MAJ) == {0, 4, 7}
    assert _tone_pcs(A_MIN) == {9, 0, 4}


def test_chord_at_returns_containing_segment_else_none():
    chords = [C_MAJ, A_MIN]
    assert _chord_at(chords, 0.5) is C_MAJ
    assert _chord_at(chords, 1.5) is A_MIN
    assert _chord_at(chords, 5.0) is None


def test_snap_moves_pitch_to_nearest_chord_tone():
    # D=62 (pc 2) in C major -> nearest chord tone is E=64 (pc 4) or C=60 (pc 0), tie -> lower delta wins C? both delta 2
    # implementation checks pitch-d then pitch+d for d=0..12, so pitch-2=60 (C, a chord tone) returns first
    assert _snap(62, C_MAJ) == 60
    # F=65 (pc 5) -> nearest is E=64 (pc4, delta1) before G=67
    assert _snap(65, C_MAJ) == 64
    # already a chord tone -> unchanged
    assert _snap(67, C_MAJ) == 67


from chiptune.score import NoteEvent, Role
from chiptune.arrange.comp_select import _importance, _lead_active, _voice_lead


def _h(pitch, start, end, vel=80):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=vel, role=Role.HARMONY)


def test_importance_rewards_duration_loudness_and_chord_fit():
    long_intone = _h(60, 0.0, 1.0, vel=100)   # C, in C major, long, loud
    short_offtone = _h(61, 0.0, 0.1, vel=40)  # C#, off-chord, short, quiet
    assert _importance(long_intone, C_MAJ) > _importance(short_offtone, C_MAJ)


def test_importance_off_chord_is_penalized_vs_same_note_in_chord():
    note = _h(64, 0.0, 0.5)  # E
    assert _importance(note, C_MAJ) > _importance(note, A_MIN.__class__(start=0.0, end=1.0, root=1, is_minor=False))


def test_lead_active_detects_overlap():
    lead = [NoteEvent(pitch=72, start=0.4, end=0.9, velocity=90, role=Role.LEAD)]
    assert _lead_active(lead, 0.5, 0.6) is True
    assert _lead_active(lead, 1.0, 1.2) is False


def test_voice_lead_shifts_octaves_toward_previous():
    # pitch 72 (C5), prev 60 (C4) -> shift down to 60
    assert _voice_lead(72, 60) == 60
    # pitch 48 (C3), prev 67 (G4) -> nearest C to G4 is C5=72 (|72-67|=5 < |60-67|=7),
    # not C4=60 as the plan's own comment claimed; corrected per the design doc
    # (docs/superpowers/specs/2026-07-22-note-selection-comp-design.md:53, "prefer the
    # one closest to the previous emitted comp note") and the verbatim algorithm's own
    # docstring ("as close as possible"). See batchA-report.md for the brief-defect note.
    assert _voice_lead(48, 67) == 72
    # no previous -> unchanged
    assert _voice_lead(64, None) == 64
