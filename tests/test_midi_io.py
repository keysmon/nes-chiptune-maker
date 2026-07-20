# tests/test_midi_io.py
from pathlib import Path

import mido
import pretty_midi
import pytest

from chiptune.score import Role, Percussion
from chiptune.midi_io import load_midi

ASSET = Path(__file__).resolve().parents[1] / "assets" / "test_theme.mid"


def test_loads_all_four_roles(tmp_path):
    score = load_midi(ASSET)
    roles = {n.role for n in score.notes}
    assert roles == {Role.LEAD, Role.HARMONY, Role.BASS, Role.PERCUSSION}


def test_tempo_is_read_from_the_file():
    score = load_midi(ASSET)
    assert score.tempo.bpm == pytest.approx(120.0, abs=0.5)


def test_percussion_notes_carry_a_kind():
    score = load_midi(ASSET)
    perc = [n for n in score.notes if n.role is Role.PERCUSSION]
    assert perc, "test asset must contain drums"
    assert all(n.percussion is not None for n in perc)
    assert {n.percussion for n in perc} >= {Percussion.KICK, Percussion.SNARE}


def test_asset_contains_an_over_budget_passage():
    """Spec 6.4: the dense section is what exercises reduction logic."""
    score = load_midi(ASSET)
    harmony = [n for n in score.notes if n.role is Role.HARMONY]
    # find the max simultaneous harmony notes
    boundaries = sorted({n.start for n in harmony})
    max_simul = max(
        sum(1 for n in harmony if n.start <= t < n.end) for t in boundaries
    )
    assert max_simul >= 4, (
        f"harmony peaks at {max_simul} simultaneous notes; the over-budget "
        "section must exceed the 1-note-per-channel budget by a clear margin"
    )


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_midi(tmp_path / "nope.mid")


def test_asset_contains_an_unclipped_sustained_lead_note():
    """Spec 6.4: the over-budget section's sustained lead note must survive
    MIDI round-tripping intact so it actually overlaps the fast-moving line
    beneath it. A same-pitch note elsewhere on the lead track would cut this
    note's sustain short via a stray note-off - see make_test_midi.py."""
    score = load_midi(ASSET)
    lead = [n for n in score.notes if n.role is Role.LEAD]
    bars_1_9_at_120bpm = 1.9 * 2.0  # 2.0s/bar at 120bpm -> 3.8s
    assert any(n.duration >= bars_1_9_at_120bpm for n in lead), (
        "expected a lead note of at least ~1.9 bars (~3.8s at 120bpm); "
        "the sustained-note scenario may have been truncated by a same-pitch collision"
    )


def test_unmapped_percussion_is_reported_not_silently_dropped(tmp_path, capsys):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))  # kick, mapped
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=49, start=0.5, end=0.6))  # crash, unmapped
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=51, start=0.7, end=0.8))  # ride, unmapped
    pm.instruments.append(drums)
    path = tmp_path / "unmapped.mid"
    pm.write(str(path))

    score = load_midi(path)
    err = capsys.readouterr().err
    assert "49" in err and "51" in err and "2" in err
    assert all(n.pitch not in (49, 51) for n in score.notes)


def test_multiple_tempo_changes_warn_and_use_the_first(tmp_path, capsys):
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120.0), time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(140.0), time=0))
    track.append(mido.Message("note_on", note=62, velocity=100, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=480))
    path = tmp_path / "tempo_change.mid"
    mid.save(str(path))

    score = load_midi(path)
    err = capsys.readouterr().err
    assert "tempo" in err.lower()
    assert score.tempo.bpm == pytest.approx(120.0, abs=0.5)
