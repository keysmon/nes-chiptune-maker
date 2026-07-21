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


VALID_ANALYSIS = (
    "[analysis]\ninclude_vocals=true\nvocal_fmin=80.0\nvocal_fmax=1000.0\n"
    "min_note_seconds=0.033\nkick_band_hz=150.0\nhat_band_hz=6000.0\n"
    "kick_low_frac_min=0.5\nhat_high_frac_min=0.15\n"
    "onset_backtrack=true\nharmony_declash=true\ndeclash_semitones=1\n"
)


def test_rejects_invalid_duty(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\nvelocity_floor=0.35\n"
        "reattack_gap=0.02\n"
        f"[pulse1]\nduty=0.33\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        f"{VALID_ANALYSIS}"
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


def test_drum_priority_loads_from_config():
    """Collision priority is a taste value, so it comes from config, not code."""
    drums = load_config().drums
    assert drums["kick"].priority > drums["snare"].priority > drums["hat"].priority


def test_drum_rejects_volume_above_four_bits():
    from chiptune.config import DrumVoice
    with pytest.raises(ValueError, match="0-15"):
        DrumVoice(period_index=4, mode="long", volume=16, frames=4, priority=3)


def test_drum_rejects_non_positive_frames():
    from chiptune.config import DrumVoice
    with pytest.raises(ValueError, match="frames"):
        DrumVoice(period_index=4, mode="long", volume=10, frames=0, priority=3)


def test_analysis_section_loads():
    cfg = load_config()
    assert cfg.analysis.include_vocals is True
    assert cfg.analysis.vocal_fmin < cfg.analysis.vocal_fmax
    assert cfg.analysis.kick_band_hz < cfg.analysis.hat_band_hz
    assert 0 < cfg.analysis.kick_low_frac_min <= 1
    assert 0 < cfg.analysis.hat_high_frac_min <= 1
    assert cfg.analysis.min_note_seconds > 0
    assert cfg.analysis.harmony_declash is True


def test_rejects_unknown_analysis_key(tmp_path):
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
        "[analysis]\ninclude_vocals=true\nvocal_fmin=80.0\nvocal_fmax=1000.0\n"
        "min_note_seconds=0.033\nkick_band_hz=150.0\nhat_band_hz=6000.0\n"
        "kick_low_frac_min=0.5\nhat_high_frac_min=0.15\n"
        "onset_backtrack=true\nharmony_declash=true\ndeclash_semitones=1\n"
        "vocal_fmn=80.0\n"
    )
    with pytest.raises(ValueError, match="vocal_fmn"):
        load_config(bad)
