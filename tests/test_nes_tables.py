import pytest
from chiptune.nes.tables import (
    CPU_NTSC, NOISE_PERIODS, PERIOD_MAX,
    midi_to_hz, pulse_period, pulse_period_to_hz,
    triangle_period, triangle_period_to_hz,
    playable_on_pulse, playable_on_triangle,
)


def test_midi_to_hz_a440():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(60) == pytest.approx(261.6256, abs=1e-3)


def test_pulse_period_round_trips_within_a_quarter_tone():
    for pitch in range(33, 108):          # A1 to B7, the usable pulse range
        f = midi_to_hz(pitch)
        p = pulse_period(f)
        back = pulse_period_to_hz(p)
        cents = 1200.0 * abs(back / f - 1.0)
        assert cents < 50.0, f"pitch {pitch} off by {cents:.1f} cents"


def test_triangle_is_one_octave_below_pulse_for_the_same_period():
    p = 200
    assert pulse_period_to_hz(p) == pytest.approx(2 * triangle_period_to_hz(p))


def test_periods_never_exceed_the_eleven_bit_register():
    for pitch in range(0, 128):
        assert pulse_period(midi_to_hz(pitch)) <= PERIOD_MAX
        assert triangle_period(midi_to_hz(pitch)) <= PERIOD_MAX


def test_there_are_sixteen_noise_periods():
    assert len(NOISE_PERIODS) == 16
    assert NOISE_PERIODS[0] == 4
    assert NOISE_PERIODS[-1] == 4068


def test_playability_matches_the_period_limits():
    assert not playable_on_pulse(20)      # too low: period would exceed 2047
    assert playable_on_pulse(60)
    assert playable_on_triangle(36)
    # NOTE: originally `assert not playable_on_triangle(120)` with the comment
    # "too high: period collapses below 1". That claim is false for this hardware.
    # Triangle's period only drops below 1 above CPU_NTSC / 64 ~= 27965 Hz, which
    # needs MIDI pitch ~141 - outside 0-127 entirely. MIDI's own ceiling (pitch 127,
    # ~12544 Hz) only reaches period ~3.46, still comfortably >= 1. So no valid MIDI
    # pitch can ever be "too high" for the triangle channel under NTSC timing;
    # pitch 120 (period ~5.68) is genuinely playable, hence the flipped assertion.
    assert playable_on_triangle(120)


def test_cpu_clock_is_ntsc():
    assert CPU_NTSC == 1789773
