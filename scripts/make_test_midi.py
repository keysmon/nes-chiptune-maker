# scripts/make_test_midi.py
"""Generate the hand-authored Phase 1 validation MIDI.

Two deliberate sections (spec 6.4):
  bars 1-4  happy path  - unambiguous lead, simple triads, bass, basic beat
  bars 5-8  over-budget - 5-note chord clusters, sustained lines overlapping
                          moving ones, lead and bass both active

The second section is what exercises note-stealing, priority resolution, and
arpeggiation under pressure. Clean input never reaches that code.
"""
from pathlib import Path

import pretty_midi

BPM = 120.0
BEAT = 60.0 / BPM          # 0.5 s
BAR = 4 * BEAT             # 2.0 s
OUT = Path(__file__).resolve().parents[1] / "assets" / "test_theme.mid"


def main() -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    lead = pretty_midi.Instrument(program=80, name="lead")
    harmony = pretty_midi.Instrument(program=81, name="harmony")
    bass = pretty_midi.Instrument(program=38, name="bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

    def add(inst, pitch, start, dur, vel=100):
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=start, end=start + dur))

    # ---- Section A: happy path, bars 1-4 -------------------------------
    melody = [72, 74, 76, 77, 76, 74, 72, 71]      # C5 D5 E5 F5 E5 D5 C5 B4
    for i, p in enumerate(melody):
        add(lead, p, i * BEAT, BEAT * 0.9)

    triads = [(60, 64, 67), (57, 60, 64), (65, 69, 72), (67, 71, 74)]
    for bar, chord in enumerate(triads):
        for p in chord:
            add(harmony, p, bar * BAR, BAR * 0.95, vel=80)

    for bar, root in enumerate([36, 33, 41, 43]):
        add(bass, root, bar * BAR, BEAT * 1.8, vel=110)
        add(bass, root, bar * BAR + 2 * BEAT, BEAT * 1.8, vel=110)

    for beat in range(16):
        t = beat * BEAT
        if beat % 4 in (0, 2):
            add(drums, 36, t, 0.1, vel=120)         # kick
        if beat % 4 == 2:
            add(drums, 38, t, 0.1, vel=110)         # snare
        add(drums, 42, t, 0.05, vel=70)             # closed hat
        add(drums, 42, t + BEAT / 2, 0.05, vel=55)

    # ---- Section B: over-budget, bars 5-8 ------------------------------
    off = 4 * BAR

    # 5-note clusters held across whole bars: far past the 1-note channel budget
    clusters = [
        (60, 64, 67, 71, 74),
        (58, 62, 65, 69, 72),
        (60, 63, 67, 70, 74),
        (59, 62, 66, 69, 73),
    ]
    for bar, chord in enumerate(clusters):
        for p in chord:
            add(harmony, p, off + bar * BAR, BAR * 0.98, vel=85)

    # Lead: sustained note overlapping a fast moving line, so a sustained voice
    # must be stolen mid-note and the hysteresis path is reachable. Pitch 84 is
    # used (not 76-80) so the moving line below never emits it and cuts the
    # sustain short with its own note-off.
    add(lead, 84, off, BAR * 2, vel=105)                     # long sustain, 2 bars
    for i in range(16):
        add(lead, 76 + (i % 5), off + BAR + i * (BEAT / 4), BEAT / 4 * 0.9, vel=95)

    # Bass: two simultaneous notes, forcing lowest-wins reduction on triangle.
    for bar in range(4):
        add(bass, 36 - bar, off + bar * BAR, BAR * 0.95, vel=115)
        add(bass, 48 - bar, off + bar * BAR, BAR * 0.95, vel=100)

    # Drums: kick and snare landing on the same tick, forcing noise-channel priority.
    for beat in range(16):
        t = off + beat * BEAT
        add(drums, 36, t, 0.1, vel=120)
        add(drums, 38, t, 0.1, vel=115)
        add(drums, 42, t + BEAT / 2, 0.05, vel=60)

    pm.instruments.extend([lead, harmony, bass, drums])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(OUT))
    print(f"wrote {OUT} ({pm.get_end_time():.2f}s, {sum(len(i.notes) for i in pm.instruments)} notes)")


if __name__ == "__main__":
    main()
