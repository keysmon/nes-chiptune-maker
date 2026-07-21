import pytest
from chiptune.score import Role, Percussion, TempoGrid
from chiptune.arrange.notation import parse_arrangement, NotationError

GRID = TempoGrid(bpm=120.0, offset=0.0, beats_per_bar=4)   # 0.5 s/beat
OCT = {"LEAD": 4, "HARM": 3, "BASS": 2}

def test_lead_degrees_resolve_in_key():
    # Key C major: degree 1 -> C, 3 -> E, 5 -> G at octave 4 (C4=60)
    text = "KEY: C maj\nLEAD: 1:1 3:1 5:2"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert [n.pitch for n in notes] == [60, 64, 67]
    assert notes[0].start == 0.0 and notes[0].end == pytest.approx(0.5)  # 1 beat
    assert notes[2].end == pytest.approx(2.0)  # 0.5+0.5+ (2 beats=1.0) -> ends at 2.0

def test_minor_key_and_accidentals_and_octave_marks():
    text = "KEY: A min\nBASS: 1:1 1,:1 b3:1 3:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.BASS]
    # A minor, bass octave 2: degree1 = 12*3+9 = 45 (A2); 1, = 33 (down an octave);
    # degrees are mode-relative (the grammar indexes the *minor* scale array when the
    # key is minor), so plain "3" is already the natural minor 3rd: 12*3+9+3 = 48 (C3).
    # "b3" flattens that by one more semitone -> scale_offset 2 -> 12*3+9+2 = 47 (B2),
    # a diminished 3rd - unusual but the correct mode-relative reading.
    assert [n.pitch for n in notes] == [45, 33, 47, 48]

def test_rests_advance_time_without_a_note():
    text = "KEY: C maj\nLEAD: R:1 1:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert len(notes) == 1 and notes[0].start == pytest.approx(0.5)  # after a 1-beat rest

def test_drums_map_to_percussion_kinds():
    text = "KEY: C maj\nDRUMS: K:1 S:1 H:1 KH:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.PERCUSSION]
    kinds = [n.percussion for n in notes]
    assert Percussion.KICK in kinds and Percussion.SNARE in kinds and Percussion.HAT in kinds
    # "KH" emits two simultaneous hits
    assert sum(1 for n in notes if n.start == pytest.approx(1.5)) == 2

def test_malformed_tokens_are_dropped_not_raised():
    text = "KEY: C maj\nLEAD: 1:1 garbage 9:1 3:x 5:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert [n.pitch for n in notes] == [60, 67]  # only 1 and 5 survive

def test_no_usable_content_raises_for_fallback():
    with pytest.raises(NotationError):
        parse_arrangement("KEY: C maj\nLEAD: all garbage here", GRID, OCT)
    with pytest.raises(NotationError):
        parse_arrangement("no key line at all\nLEAD: 1:1", GRID, OCT)

@pytest.mark.parametrize("bad", ["inf", "nan", "1e999", "-inf"])
def test_nonfinite_durations_are_dropped_not_raised(bad):
    # inf/nan clear the `dur <= 0` guard (nan compares False, inf > 0) and would make
    # a non-finite NoteEvent end -> ValueError, which escapes NotationError. Must drop.
    text = f"KEY: C maj\nLEAD: 3:{bad} 1:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert [n.pitch for n in notes] == [60]  # bad-duration token dropped, 1:1 survives

def test_absurd_finite_duration_is_dropped_not_rendered():
    # A finite-but-enormous duration clears the isfinite guard but would size the
    # render buffer to gigabytes and OOM the synth OUTSIDE the fallback. Must drop.
    text = "KEY: C maj\nLEAD: 1:100000 5:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert [n.pitch for n in notes] == [67]      # 1:100000 dropped, 5:1 (G4) survives
    assert all(n.end <= 600.0 for n in notes)    # nothing beyond the safety rail reaches the synth

def test_max_seconds_truncates_a_runaway_voice():
    # A voice that runs far past the song length is truncated to the song-relative
    # cap, not the 600s rail. GRID is 0.5 s/beat, so 5 notes of 2 beats end at
    # 1/2/3/4/5 s; max_seconds=4.0 drops the note ending at 5 s.
    text = "KEY: C maj\nBASS: 1:2 1:2 1:2 1:2 1:2"
    notes = [n for n in parse_arrangement(text, GRID, OCT, max_seconds=4.0) if n.role is Role.BASS]
    assert notes and all(n.end <= 4.0 for n in notes)
    assert max(n.end for n in notes) == pytest.approx(4.0)  # kept everything up to the cap

def test_harmony_voice_maps_to_harmony_role():
    text = "KEY: C maj\nHARM: 1:1 3:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.HARMONY]
    # HARM octave 3: degree 1 = 12*4+0 = 48 (C3), degree 3 = 52 (E3)
    assert [n.pitch for n in notes] == [48, 52]

def test_voice_split_across_lines_concatenates():
    # A long part the LLM wraps onto a second same-voice line must append, not clobber.
    text = "KEY: C maj\nLEAD: 1:1\nLEAD: 5:1"
    notes = [n for n in parse_arrangement(text, GRID, OCT) if n.role is Role.LEAD]
    assert [n.pitch for n in notes] == [60, 67]
    assert notes[1].start == pytest.approx(0.5)  # 2nd note follows the 1st (not overwritten)
