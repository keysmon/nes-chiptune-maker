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
