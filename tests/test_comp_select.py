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
