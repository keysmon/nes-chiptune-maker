"""Band-limited pulse oscillator using mipmapped wavetables.

A naive square wave made by thresholding a sine aliases badly at audio sample
rates, which sounds like harsh digital grit rather than a clean chip tone
(spec Risk R4). Instead we precompute one wavetable per octave per duty, each
holding only the harmonics that stay under Nyquist for the top of that
octave, then read it with phase accumulation and linear interpolation. That
is O(1) per sample.

Fourier series of a pulse of duty d (DC term dropped so output is zero-mean):
    p(t) = sum_{n>=1} (2 / (n*pi)) * sin(n*pi*d) * cos(2*pi*n*t)
"""
from __future__ import annotations

import numpy as np

from ..config import VALID_DUTIES

# One table per octave, covering MIDI 0 (8.18 Hz) through MIDI 127 (12543 Hz).
N_OCTAVES = 11
BASE_HZ = 8.1758  # MIDI note 0


class PulseBank:
    def __init__(self, sample_rate: int, table_size: int = 2048):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive (got {sample_rate})")
        self.sample_rate = sample_rate
        self.table_size = table_size
        self._raw: dict[tuple[float, int], np.ndarray] = {}
        self._tables: dict[tuple[float, int], np.ndarray] = {}
        self._gain: float | None = None

    def _octave_index(self, freq_hz: float) -> int:
        if freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive (got {freq_hz})")
        idx = int(np.floor(np.log2(freq_hz / BASE_HZ)))
        return max(0, min(N_OCTAVES - 1, idx))

    def _raw_table(self, duty: float, octave: int) -> np.ndarray:
        """Un-normalized band-limited pulse table for one (duty, octave) band."""
        key = (duty, octave)
        cached = self._raw.get(key)
        if cached is not None:
            return cached

        # Harmonics safe for the highest frequency in this octave band.
        top_hz = BASE_HZ * (2.0 ** (octave + 1))
        nyquist = self.sample_rate / 2.0
        n_harmonics = max(1, int(nyquist // top_hz))

        t = np.arange(self.table_size, dtype=np.float64) / self.table_size
        table = np.zeros(self.table_size, dtype=np.float64)
        for n in range(1, n_harmonics + 1):
            amp = (2.0 / (n * np.pi)) * np.sin(n * np.pi * duty)
            table += amp * np.cos(2.0 * np.pi * n * t)

        self._raw[key] = table
        return table

    def _global_gain(self) -> float:
        """One scalar for the whole bank: 1 / (max peak over every valid table).

        A pulse's fundamental amplitude is (2/pi)*sin(pi*duty) and does NOT depend
        on octave, so the note's perceived loudness is already constant across the
        mip pyramid. What differs is the overall peak: high-octave tables keep only
        the harmonics under Nyquist (as few as one), so their peak is lower even
        though the fundamental is identical. Normalizing each table to its OWN peak
        would therefore divide the high octaves by a smaller number and scale their
        fundamental *up* - a ~12 dB loudness inversion for narrow duties. A single
        global gain instead rescales the whole bank uniformly: it guarantees no
        table clips (the loudest just reaches 1.0) while leaving the fundamentals
        constant. High notes stay legitimately thinner - that is authentic NES
        band-limiting, not a defect to equalize away.
        """
        if self._gain is None:
            peak = 0.0
            for duty in VALID_DUTIES:
                for octave in range(N_OCTAVES):
                    peak = max(peak, float(np.abs(self._raw_table(duty, octave)).max()))
            self._gain = 1.0 / peak if peak > 0.0 else 1.0
        return self._gain

    def _table(self, duty: float, octave: int) -> np.ndarray:
        key = (duty, octave)
        cached = self._tables.get(key)
        if cached is not None:
            return cached

        table = self._raw_table(duty, octave) * self._global_gain()
        self._tables[key] = table
        return table

    def render(
        self,
        freq_hz: float,
        duty: float,
        n_samples: int,
        phase: float,
    ) -> tuple[np.ndarray, float]:
        """Return (samples, next_phase). `phase` is in turns, i.e. [0, 1)."""
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float64), phase

        table = self._table(duty, self._octave_index(freq_hz))
        step = freq_hz / self.sample_rate
        phases = phase + step * np.arange(n_samples, dtype=np.float64)

        pos = (phases % 1.0) * self.table_size
        i0 = pos.astype(np.int64)
        frac = pos - i0
        i1 = (i0 + 1) % self.table_size
        out = table[i0] * (1.0 - frac) + table[i1] * frac

        return out, float((phase + step * n_samples) % 1.0)
