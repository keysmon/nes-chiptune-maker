"""Unit tests for the phantom-echo arranger (pure, no I/O)."""
import pytest

from chiptune.arrange.phantom_echo import add_phantom_echo
from chiptune.config import EchoConfig
from chiptune.score import NoteEvent, Role

FR = 60.0  # frame rate; delay_frames=4 -> 4/60 ~= 0.0667s


def _echo(**over):
    base = dict(enabled=True, delay_frames=4, volume=0.5, min_lead_seconds=0.12)
    base.update(over)
    return EchoConfig(**base)


def _lead(pitch, start, end, vel=100):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=vel, role=Role.LEAD)


def _harm(pitch, start, end, vel=80):
    return NoteEvent(pitch=pitch, start=start, end=end, velocity=vel, role=Role.HARMONY)


def _is_monophonic(notes):
    ordered = sorted(notes, key=lambda n: n.start)
    return all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))


def test_long_lead_note_echoes_into_an_empty_comp():
    lead = [_lead(72, 0.0, 0.5, vel=100)]
    out = add_phantom_echo(lead, [], _echo(), FR)
    assert len(out) == 1
    e = out[0]
    assert e.pitch == 72
    assert e.role is Role.HARMONY
    assert e.velocity == 50            # round(100 * 0.5)
    assert e.start == pytest.approx(4 / FR)
    assert e.end == pytest.approx(0.5 + 4 / FR)


def test_short_lead_note_gets_no_echo():
    lead = [_lead(72, 0.0, 0.10)]      # 0.10 <= min_lead_seconds (0.12)
    out = add_phantom_echo(lead, [], _echo(), FR)
    assert out == []


def test_comp_wins_echo_clamped_to_the_gap_after_a_comp_note():
    lead = [_lead(72, 0.0, 0.5)]
    comp = [_harm(60, 0.0, 0.3)]       # occupies 0.0-0.3
    out = sorted(add_phantom_echo(lead, comp, _echo(), FR), key=lambda n: n.start)
    assert comp[0] in out              # comp note preserved unchanged
    echo = [n for n in out if n.pitch == 72]
    assert len(echo) == 1
    assert echo[0].start == pytest.approx(0.3)      # pushed to start after the comp note ends
    assert _is_monophonic(out)


def test_echo_fully_inside_a_comp_note_is_dropped():
    lead = [_lead(72, 0.0, 0.2)]       # 0.2 > 0.12, so an echo is generated
    comp = [_harm(60, 0.0, 1.0)]       # covers the whole delayed echo span
    out = add_phantom_echo(lead, comp, _echo(), FR)
    assert out == comp                 # only the comp note survives
    assert all(n.pitch != 72 for n in out)


def test_echo_fills_the_gap_between_two_comp_notes_both_preserved():
    lead = [_lead(72, 0.0, 0.5)]
    comp = [_harm(60, 0.0, 0.1), _harm(64, 0.2, 0.3)]   # gap = [0.1, 0.2]
    out = sorted(add_phantom_echo(lead, comp, _echo(), FR), key=lambda n: n.start)
    assert comp[0] in out and comp[1] in out            # both comp notes preserved
    echo = [n for n in out if n.pitch == 72]
    assert len(echo) == 1
    assert echo[0].start == pytest.approx(0.1)
    assert echo[0].end == pytest.approx(0.2)            # clamped to end before the next comp note
    assert _is_monophonic(out)


def test_two_overlapping_echoes_are_made_monophonic():
    lead = [_lead(72, 0.0, 0.3), _lead(74, 0.05, 0.35)]  # both > 0.12
    out = add_phantom_echo(lead, [], _echo(), FR)
    assert len(out) == 2
    assert _is_monophonic(out)


def test_empty_lead_returns_the_comp_unchanged():
    comp = [_harm(60, 0.0, 0.3)]
    out = add_phantom_echo([], comp, _echo(), FR)
    assert out == comp


def test_works_with_multi_note_arp_style_comp_input():
    # A denser (arp-style) monophonic comp: the echo still only lands in a real gap.
    lead = [_lead(72, 0.0, 1.0)]
    comp = [_harm(60, 0.0, 0.2), _harm(64, 0.2, 0.4), _harm(67, 0.4, 0.6)]  # gap starts at 0.6
    out = sorted(add_phantom_echo(lead, comp, _echo(), FR), key=lambda n: n.start)
    for c in comp:
        assert c in out
    echo = [n for n in out if n.pitch == 72]
    assert len(echo) == 1
    assert echo[0].start == pytest.approx(0.6)          # first clear gap
    assert _is_monophonic(out)
