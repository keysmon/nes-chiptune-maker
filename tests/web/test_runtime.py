"""Pure-logic tests for the web runtime (no server needed)."""
from chiptune.config import config_from_overrides
from chiptune.web.app import expand_paths
from chiptune.web.runtime import _analysis_signature
from chiptune.web.schema import schema_with_defaults


def test_expand_paths_nests_dot_paths():
    flat = {"arrange.chord_octave": 4, "drums.snare.volume": 9, "noise_lowpass_hz": 4000}
    assert expand_paths(flat) == {
        "arrange": {"chord_octave": 4},
        "drums": {"snare": {"volume": 9}},
        "noise_lowpass_hz": 4000,
    }


def test_config_from_overrides_applies_and_validates():
    cfg = config_from_overrides({"levels": {"noise": 0.3}, "vibrato": {"depth_semitones": 0.9}})
    assert cfg.levels["noise"] == 0.3
    assert cfg.vibrato.depth_semitones == 0.9
    # untouched defaults survive
    assert cfg.pulse1.volume == config_from_overrides({}).pulse1.volume


def test_synthesis_change_keeps_analysis_signature_but_analysis_change_differs():
    base = config_from_overrides({})
    synth = config_from_overrides({"levels": {"noise": 0.2}, "vibrato": {"rate_hz": 3.0}})
    analysis = config_from_overrides({"arrange": {"harmony_mode": "transcribe"}})
    assert _analysis_signature(synth) == _analysis_signature(base), "synthesis change must not trigger re-analyze"
    assert _analysis_signature(analysis) != _analysis_signature(base), "analysis change must trigger re-analyze"


def test_schema_controls_have_valid_paths_and_defaults():
    for c in schema_with_defaults():
        assert c["default"] is not None, f"{c['path']} has no default"
        assert c["type"] in ("range", "toggle", "choice")


def test_every_arrange_field_is_classified_analysis_or_synthesis():
    """Guard: adding an [arrange] field forces classifying it as Score-affecting or
    synthesis-only, so a new harmony/arrangement knob can't silently skip re-analyze."""
    from dataclasses import fields
    from chiptune.config import ArrangeConfig
    from chiptune.web.runtime import _ANALYSIS_ARRANGE_KEYS, _SYNTHESIS_ARRANGE_KEYS
    all_fields = {f.name for f in fields(ArrangeConfig)}
    classified = _ANALYSIS_ARRANGE_KEYS | _SYNTHESIS_ARRANGE_KEYS
    assert all_fields == classified, (
        f"unclassified arrange fields: {all_fields ^ classified} - add each to "
        "_ANALYSIS_ARRANGE_KEYS or _SYNTHESIS_ARRANGE_KEYS in web/runtime.py"
    )


def test_echo_change_flips_the_analysis_signature():
    on = config_from_overrides({"echo": {"enabled": True}})
    off = config_from_overrides({"echo": {"enabled": False}})
    assert _analysis_signature(on) != _analysis_signature(off), (
        "toggling [echo] must re-analyze (it changes the Score's HARMONY list)"
    )


def test_echo_field_change_flips_the_analysis_signature():
    # Not just `enabled`: every EchoConfig field is hashed (via vars(cfg.echo)) and
    # is analysis-affecting (it changes the echo notes), so a non-enabled field
    # change must also re-analyze. Locks against a refactor that hashes only `enabled`.
    base = config_from_overrides({"echo": {"enabled": True}})
    slower = config_from_overrides({"echo": {"enabled": True, "delay_frames": 8}})
    assert _analysis_signature(slower) != _analysis_signature(base), (
        "an echo delay_frames change must re-analyze (it changes the Score's echo notes)"
    )


def test_synthesis_change_still_keeps_signature_stable_with_echo_in_the_blob():
    base = config_from_overrides({})
    synth = config_from_overrides({"levels": {"noise": 0.2}})
    assert _analysis_signature(synth) == _analysis_signature(base), (
        "a synthesis-only change must not re-analyze even though echo is now hashed"
    )
