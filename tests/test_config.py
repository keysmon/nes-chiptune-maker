# tests/test_config.py
import pytest
from chiptune.config import load_config, DEFAULT_CONFIG_PATH, config_from_dict, default_raw_config


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


VALID_VIBRATO = (
    "[vibrato]\nrate_hz=6.0\ndepth_semitones=0.25\ndelay_frames=12\nenabled=true\n"
)


VALID_AI = (
    "[ai]\nbase_url=\"https://api.groq.com/openai/v1\"\n"
    "model=\"llama-3.3-70b-versatile\"\napi_key_env=\"GROQ_API_KEY\"\n"
    "temperature=0.4\nmax_tokens=4000\n"
)


VALID_ARRANGE = (
    "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
    "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
    "borrow_idle_frames=30\nborrow_hysteresis_frames=15\nvelocity_floor=0.35\n"
    "reattack_gap=0.02\nharmony_mode=\"chords\"\nchord_comp_pattern=\"up\"\n"
    "chord_subdivision=2\nchord_octave=4\nchord_tones=3\nchord_smooth_beats=2\n"
    "melody_min_seconds=0.08\nbass_min_seconds=0.08\n"
    "harmony_rest_on_busy_melody=false\n"
)


def test_rejects_invalid_duty(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        f"{VALID_ARRANGE}"
        f"[pulse1]\nduty=0.33\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        f"{VALID_ANALYSIS}"
        f"{VALID_VIBRATO}"
        f"{VALID_AI}"
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


def test_arrange_section_loads_sparse_arranger_keys():
    cfg = load_config()
    assert cfg.arrange.harmony_mode == "chords"
    assert cfg.arrange.chord_comp_pattern == "up"
    assert cfg.arrange.chord_subdivision == 2
    assert cfg.arrange.chord_octave == 4
    assert cfg.arrange.chord_tones == 3
    assert cfg.arrange.chord_smooth_beats == 4
    assert cfg.arrange.melody_min_seconds == pytest.approx(0.08)
    assert cfg.arrange.bass_min_seconds == pytest.approx(0.08)
    assert cfg.arrange.harmony_rest_on_busy_melody is False


def test_rejects_unknown_arrange_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\nvelocity_floor=0.35\n"
        "reattack_gap=0.02\nharmony_mode=\"chords\"\nharmnoy_mode=\"chords\"\n"
        f"[pulse1]\nduty=0.5\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        f"{VALID_ANALYSIS}"
        f"{VALID_VIBRATO}"
    )
    with pytest.raises(ValueError, match="harmnoy_mode"):
        load_config(bad)


def test_rejects_invalid_harmony_mode(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\nvelocity_floor=0.35\n"
        "reattack_gap=0.02\nharmony_mode=\"riffs\"\nchord_comp_pattern=\"up\"\n"
        "chord_subdivision=2\nchord_octave=4\nchord_tones=3\nchord_smooth_beats=2\n"
        "melody_min_seconds=0.08\nbass_min_seconds=0.08\n"
        "harmony_rest_on_busy_melody=false\n"
        f"[pulse1]\nduty=0.5\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        f"{VALID_ANALYSIS}"
        f"{VALID_VIBRATO}"
        f"{VALID_AI}"
    )
    with pytest.raises(ValueError, match="harmony_mode"):
        load_config(bad)


def test_ai_config_loads():
    cfg = load_config()
    assert cfg.arrange.arrange_mode in ("heuristic", "ai")
    assert cfg.ai.model and cfg.ai.base_url.startswith("http")
    assert cfg.ai.api_key_env == "GROQ_API_KEY"
    assert 0.0 <= cfg.ai.temperature <= 2.0


def test_vibrato_section_loads():
    cfg = load_config()
    assert cfg.vibrato.enabled is True
    assert cfg.vibrato.rate_hz > 0
    assert cfg.vibrato.depth_semitones >= 0
    assert cfg.vibrato.delay_frames >= 0


def test_output_filter_cutoffs_load():
    cfg = load_config()
    assert cfg.output_highpass_hz == pytest.approx(30.0)
    assert cfg.output_lowpass_hz == pytest.approx(13000.0)


def test_rejects_missing_vibrato_section(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "sample_rate=44100\nframe_rate=60.0\n"
        "[arrange]\nsubdivision=16\nquantize_strength=1.0\nmin_duration=0.03\n"
        "arpeggio_frames=2\nbass_low=28\nbass_high=55\nborrow_enabled=false\n"
        "borrow_idle_frames=30\nborrow_hysteresis_frames=15\nvelocity_floor=0.35\n"
        "reattack_gap=0.02\n"
        f"[pulse1]\nduty=0.5\n{VALID_CHANNEL}"
        f"[pulse2]\nduty=0.25\n{VALID_CHANNEL}"
        f"[triangle]\nduty=0.0\n{VALID_CHANNEL}"
        f"[noise]\nduty=0.0\n{VALID_CHANNEL}"
        f"{VALID_ANALYSIS}"
    )
    with pytest.raises(ValueError, match=r"\[vibrato\]"):
        load_config(bad)


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


def test_arrange_config_has_harmony_source_and_min_gap():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.arrange.harmony_source == "select"
    assert cfg.arrange.select_min_gap == pytest.approx(0.30)


def test_invalid_harmony_source_rejected():
    raw = default_raw_config()
    raw["arrange"]["harmony_source"] = "bogus"
    with pytest.raises(ValueError, match="harmony_source"):
        config_from_dict(raw)


def test_negative_min_gap_rejected():
    raw = default_raw_config()
    raw["arrange"]["select_min_gap"] = -0.1
    with pytest.raises(ValueError, match="select_min_gap"):
        config_from_dict(raw)


def test_echo_section_loads_disabled_by_default():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.echo.enabled is False
    assert cfg.echo.delay_frames == 4
    assert cfg.echo.volume == pytest.approx(0.5)
    assert cfg.echo.min_lead_seconds == pytest.approx(0.12)


def test_echo_section_missing_falls_back_to_disabled_defaults():
    raw = default_raw_config()
    del raw["echo"]
    cfg = config_from_dict(raw)
    assert cfg.echo.enabled is False
    assert cfg.echo.delay_frames == 4


def test_echo_rejects_delay_frames_below_one():
    raw = default_raw_config()
    raw["echo"]["delay_frames"] = 0
    with pytest.raises(ValueError, match="delay_frames"):
        config_from_dict(raw)


def test_echo_rejects_volume_out_of_range():
    raw = default_raw_config()
    raw["echo"]["volume"] = 1.5
    with pytest.raises(ValueError, match="volume"):
        config_from_dict(raw)


def test_echo_rejects_negative_min_lead_seconds():
    raw = default_raw_config()
    raw["echo"]["min_lead_seconds"] = -0.01
    with pytest.raises(ValueError, match="min_lead_seconds"):
        config_from_dict(raw)


def test_echo_rejects_unknown_key():
    raw = default_raw_config()
    raw["echo"]["delayy_frames"] = 4
    with pytest.raises(ValueError, match="delayy_frames"):
        config_from_dict(raw)


def test_echo_accepts_boundary_values():
    # Accept-at-boundary: locks the validator bounds (< vs <=) so an off-by-one
    # in EchoConfig.__post_init__ would fail a test, not slip through.
    from chiptune.config import EchoConfig
    assert EchoConfig(delay_frames=1).delay_frames == 1              # delay_frames >= 1
    assert EchoConfig(volume=0.0).volume == 0.0                      # volume >= 0.0
    assert EchoConfig(volume=1.0).volume == 1.0                      # volume <= 1.0
    assert EchoConfig(min_lead_seconds=0.0).min_lead_seconds == 0.0  # min_lead_seconds >= 0
