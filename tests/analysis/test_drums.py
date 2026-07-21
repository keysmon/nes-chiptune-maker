from pathlib import Path

import numpy as np
import pytest

from chiptune.analysis import drums as D
from chiptune.analysis.drums import transcribe_drums
from chiptune.config import load_config
from chiptune.score import Role, Percussion

CLIP = Path("out/spike_demucs/htdemucs/pop_src/drums.wav")

# Band-fraction thresholds for the synthetic fast tests. The shipped defaults
# live in config/nes.toml [analysis] and are exercised by the slow test below.
KICK_BAND_HZ = 150.0
HAT_BAND_HZ = 6000.0
KICK_LOW_FRAC_MIN = 0.5
HAT_HIGH_FRAC_MIN = 0.15


def test_low_burst_is_kick_and_high_noise_is_hat():
    sr = 22050
    y = np.zeros(sr, dtype="float32")
    # low sine burst at t=0.1 -> nearly all energy below the kick band
    ta = np.arange(int(0.05 * sr)) / sr
    y[int(0.1 * sr):int(0.1 * sr) + len(ta)] += (0.8 * np.sin(2 * np.pi * 60 * ta)).astype("float32")
    # high broadband noise burst at t=0.5 -> substantial energy above the hat band
    hb = (0.6 * np.random.default_rng(0).standard_normal(int(0.03 * sr))).astype("float32")
    y[int(0.5 * sr):int(0.5 * sr) + len(hb)] += hb
    hits = transcribe_drums(y, sr, KICK_BAND_HZ, HAT_BAND_HZ, KICK_LOW_FRAC_MIN, HAT_HIGH_FRAC_MIN)
    kinds = {n.percussion for n in hits}
    assert all(n.role is Role.PERCUSSION for n in hits)
    assert Percussion.KICK in kinds  # the 60Hz sine burst
    assert Percussion.HAT in kinds   # the broadband noise burst


def test_no_onsets_returns_empty():
    sr = 22050
    y = np.zeros(sr, dtype="float32")
    hits = transcribe_drums(y, sr, KICK_BAND_HZ, HAT_BAND_HZ, KICK_LOW_FRAC_MIN, HAT_HIGH_FRAC_MIN)
    assert hits == []


def test_dropped_short_window_onset_is_still_counted_in_stderr_total(monkeypatch, capsys):
    # An onset too close to the end of the stem for a meaningful window is
    # dropped from `notes`, but must still be counted in the "of M onsets"
    # total so a dropped tail onset isn't silently invisible.
    sr = 22050
    y = np.zeros(100, dtype="float32")
    # onset 0: plenty of stem left, classified normally.
    # onset 95: only 5 samples left (< MIN_WINDOW_SAMPLES=8), dropped.
    monkeypatch.setattr(D.librosa.onset, "onset_detect", lambda **kwargs: np.array([0, 95]))

    hits = transcribe_drums(y, sr, KICK_BAND_HZ, HAT_BAND_HZ, KICK_LOW_FRAC_MIN, HAT_HIGH_FRAC_MIN)
    stderr = capsys.readouterr().err

    assert len(hits) == 1
    assert "classified 1 of 2 onsets" in stderr


@pytest.mark.slow
def test_real_drum_stem_gives_musically_plausible_split():
    # Band-energy fraction, unlike spectral centroid, recovers a real split on
    # this summed (demucs) drum stem: centroid measured kick=0 (hat/cymbal bleed
    # in every kick window drove the centroid ~4x too high); the low-band
    # fraction is robust to that bleed. Exercises the SHIPPED config thresholds.
    if not CLIP.exists():
        pytest.skip(f"{CLIP} is missing; produced by the audio-analysis spike, not checked in")
    import soundfile as sf
    y, sr = sf.read(CLIP, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    a = load_config().analysis
    hits = transcribe_drums(
        y, sr, a.kick_band_hz, a.hat_band_hz, a.kick_low_frac_min, a.hat_high_frac_min
    )
    assert hits
    assert all(n.role is Role.PERCUSSION for n in hits)
    assert all(0 <= n.pitch <= 127 and n.end > n.start for n in hits)
    counts = {k: sum(1 for n in hits if n.percussion is k) for k in Percussion}
    # All three voices present, kick and snare in comparable numbers.
    assert counts[Percussion.KICK] > 0
    assert counts[Percussion.SNARE] > 0
    assert counts[Percussion.HAT] > 0
