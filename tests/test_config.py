# tests/test_config.py
import pytest
from chiptune.config import load_config


def test_loads_default_config():
    cfg = load_config()
    assert cfg.sample_rate == 44100
    assert cfg.frame_rate == pytest.approx(60.0)
    assert cfg.arrange.subdivision == 16


def test_triangle_config_is_present_and_full_volume():
    cfg = load_config()
    assert cfg.triangle.volume == 15


def test_pulse_duty_must_be_a_real_chip_value():
    cfg = load_config()
    assert cfg.pulse1.duty in (0.125, 0.25, 0.5, 0.75)
    assert cfg.pulse2.duty in (0.125, 0.25, 0.5, 0.75)


VALID_CHANNEL = "volume=12\nattack_frames=0\ndecay_frames=0\nsustain=1.0\nrelease_frames=0\n"


def test_rejects_invalid_duty(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\n"
        f"[pulse1]\nduty=0.33\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
    )
    with pytest.raises(ValueError, match="duty"):
        load_config(bad)


def test_rejects_unknown_drum_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\n"
        f"[pulse1]\nduty=0.5\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        "[drums.hihat]\nperiod_index=2\nmode=\"long\"\nvolume=6\nframes=2\n"
    )
    with pytest.raises(ValueError, match="hihat"):
        load_config(bad)


def test_rejects_unknown_level_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\n"
        f"[pulse1]\nduty=0.5\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        "[levels]\npulse_2=0.85\n"
    )
    with pytest.raises(ValueError, match="pulse_2"):
        load_config(bad)


def test_rejects_volume_above_four_bits(tmp_path):
    cfg = load_config()
    with pytest.raises(ValueError, match="0-15"):
        type(cfg.pulse1)(duty=0.5, volume=16, attack_frames=0,
                         decay_frames=0, sustain=1.0, release_frames=0)
