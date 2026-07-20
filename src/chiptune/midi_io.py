"""Read a MIDI file into a Score."""
from __future__ import annotations

from pathlib import Path

import pretty_midi

from .score import NoteEvent, Percussion, Role, Score, TempoGrid

DEFAULT_TRACK_ROLES: dict[int, Role] = {0: Role.LEAD, 1: Role.HARMONY, 2: Role.BASS}

# General MIDI percussion key map, narrowed to the three kinds the noise channel supports.
GM_PERCUSSION: dict[int, Percussion] = {
    35: Percussion.KICK, 36: Percussion.KICK,
    38: Percussion.SNARE, 40: Percussion.SNARE,
    37: Percussion.SNARE,
    42: Percussion.HAT, 44: Percussion.HAT, 46: Percussion.HAT,
}


def load_midi(path: str | Path, role_map: dict[int, Role] | None = None) -> Score:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    pm = pretty_midi.PrettyMIDI(str(path))
    roles = DEFAULT_TRACK_ROLES if role_map is None else role_map

    tempi_times, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0

    notes: list[NoteEvent] = []
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum:
            for n in inst.notes:
                kind = GM_PERCUSSION.get(n.pitch)
                if kind is None:
                    continue  # unmapped percussion is dropped, not guessed at
                notes.append(NoteEvent(
                    pitch=n.pitch, start=float(n.start), end=float(n.end),
                    velocity=n.velocity, role=Role.PERCUSSION, percussion=kind,
                ))
            continue

        role = roles.get(idx, Role.HARMONY)
        for n in inst.notes:
            notes.append(NoteEvent(
                pitch=n.pitch, start=float(n.start), end=float(n.end),
                velocity=n.velocity, role=role,
            ))

    notes.sort(key=lambda n: (n.start, n.pitch))
    duration = max((n.end for n in notes), default=0.0)
    return Score(
        tempo=TempoGrid(bpm=bpm, offset=0.0, beats_per_bar=4),
        notes=notes,
        duration=duration,
    )
