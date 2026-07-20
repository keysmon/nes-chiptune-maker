# tests/test_midi_io.py
from pathlib import Path

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
