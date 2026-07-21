import numpy as np
import pytest
from pathlib import Path

from chiptune.analysis.separate import separate_stems, STEM_SR

CLIP = Path("out/pop_src.wav")  # 30s pop clip produced by the spike

if not CLIP.exists():
    pytest.skip(
        f"{CLIP} is missing; it is produced by the audio-analysis spike and is not "
        "checked in (out/ is gitignored) - regenerate it before running these tests",
        allow_module_level=True,
    )


@pytest.mark.slow
def test_separates_into_four_named_stems(tmp_path):
    stems = separate_stems(CLIP, cache_dir=tmp_path)
    assert set(stems) == {"drums", "bass", "other", "vocals"}
    for name, a in stems.items():
        assert a.ndim == 1 and a.dtype == np.float32
        assert len(a) / STEM_SR > 25  # ~30s clip


@pytest.mark.slow
def test_second_call_hits_cache_and_matches(tmp_path):
    a = separate_stems(CLIP, cache_dir=tmp_path)
    b = separate_stems(CLIP, cache_dir=tmp_path)  # cached
    np.testing.assert_array_equal(a["bass"], b["bass"])
