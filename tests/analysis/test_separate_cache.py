"""Fast (no-model) tests for the crash-safe stem cache.

A separation killed mid-write must never leave a truncated `.npz` occupying the
final cache path, and a corrupt cache from any earlier run must self-heal rather
than wedge the input forever. These monkeypatch the Demucs seam so they stay
fast and do not depend on the `out/` spike artifacts.
"""
import numpy as np
import soundfile as sf

from chiptune.analysis import separate as sep
from chiptune.analysis.separate import separate_stems, STEM_NAMES, STEM_SR


def _tiny_audio(path):
    sf.write(path, np.zeros(2205, dtype="float32"), STEM_SR)


def test_corrupt_cache_is_treated_as_miss(tmp_path, monkeypatch):
    audio = tmp_path / "in.wav"
    _tiny_audio(audio)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Pre-place a truncated/garbage file at the exact cache path a real run uses.
    digest = sep._content_hash(audio, sep.MODEL_NAME)
    cache_file = cache_dir / f"{digest}.npz"
    cache_file.write_bytes(b"PK\x03\x04 not a real npz -- truncated write")

    fake = {n: (np.arange(16, dtype="float32") + i) for i, n in enumerate(STEM_NAMES)}
    calls = {"n": 0}

    def fake_sep(_path):
        calls["n"] += 1
        return dict(fake)

    monkeypatch.setattr(sep, "_separate_uncached", fake_sep)

    # A corrupt cache must NOT raise; it must fall through to re-separation.
    stems = separate_stems(audio, cache_dir=cache_dir)
    assert set(stems) == set(STEM_NAMES)
    for n in STEM_NAMES:
        np.testing.assert_array_equal(stems[n], fake[n])
    assert calls["n"] == 1  # re-separated exactly once

    # It also healed: the atomic write replaced the corrupt file, so a second
    # call is a clean cache hit (no further separation) and matches.
    stems2 = separate_stems(audio, cache_dir=cache_dir)
    assert calls["n"] == 1  # cache hit -> no re-separation
    for n in STEM_NAMES:
        np.testing.assert_array_equal(stems2[n], fake[n])


def test_successful_write_leaves_only_the_final_npz(tmp_path, monkeypatch):
    audio = tmp_path / "in.wav"
    _tiny_audio(audio)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    fake = {n: np.ones(8, dtype="float32") for n in STEM_NAMES}
    monkeypatch.setattr(sep, "_separate_uncached", lambda _p: dict(fake))

    separate_stems(audio, cache_dir=cache_dir)

    files = sorted(p.name for p in cache_dir.iterdir())
    assert len(files) == 1, f"expected exactly one cache file, got {files}"
    assert files[0].endswith(".npz")
    assert not files[0].startswith(".stemtmp")  # no leftover atomic-write temp
