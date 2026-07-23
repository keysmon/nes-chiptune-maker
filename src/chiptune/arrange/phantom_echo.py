"""Phantom echo: re-use the lead melody, delayed, on the harmony channel to fill
the comp's rests so a thin lead reads fuller. Design:
docs/superpowers/specs/2026-07-22-phantom-echo-design.md

Each substantial lead note gets a single delayed copy tagged Role.HARMONY,
admitted only into the clear gaps between comp notes (the comp always wins a
collision). The merged list is kept strictly monophonic so the allocator's
arpeggiate step renders it 1:1 instead of buzz-cycling overlaps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..score import NoteEvent, Role

if TYPE_CHECKING:
    from ..config import EchoConfig


def _enforce_mono(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Sort by start and clamp each note's end to the next note's start, dropping
    any note that collapses to zero length. Guarantees strict monophony.

    Deliberately mirrors comp_select._enforce_mono; kept separate rather than
    shared so the two modules' monophony rules can evolve independently.
    """
    ordered = sorted(notes, key=lambda n: n.start)
    out: list[NoteEvent] = []
    for i, n in enumerate(ordered):
        end = n.end
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1].start)
        if end > n.start:
            out.append(NoteEvent(pitch=n.pitch, start=n.start, end=end,
                                 velocity=n.velocity, role=Role.HARMONY))
    return out


def _clamp_to_gap(echo: NoteEvent, comp: list[NoteEvent]) -> NoteEvent | None:
    """Clamp `echo` into the comp gap containing its start; return the clamped
    HARMONY note, or None if there is no room (echo fully inside/over a comp note).

    `comp` must be sorted by start. The gap is [g0, g1): g0 walks forward through
    the whole contiguous/overlapping run of comp notes covering the onset (a
    single-note lookahead would mistake two back-to-back comp notes for "no gap"),
    so g0 ends up at the end of that run. g1 is the start of the first comp note
    beginning after g0 (or echo.end if none). The comp always wins, so the echo
    only ever takes gap space.
    """
    g0 = echo.start
    g1 = echo.end
    i = 0
    while i < len(comp) and comp[i].start <= g0:
        g0 = max(g0, comp[i].end)
        i += 1
    if i < len(comp):
        g1 = min(g1, comp[i].start)
    start = max(echo.start, g0)
    end = min(echo.end, g1)
    if end <= start:
        return None
    return NoteEvent(pitch=echo.pitch, start=start, end=end,
                     velocity=echo.velocity, role=Role.HARMONY)


def add_phantom_echo(
    lead_notes: list[NoteEvent],
    harmony_notes: list[NoteEvent],
    echo_cfg: "EchoConfig",
    frame_rate: float,
) -> list[NoteEvent]:
    """Return the comp (`harmony_notes`) plus a delayed, gap-filling echo of the lead.

    For each lead note longer than `echo_cfg.min_lead_seconds`, one echo copy is
    made at `+delay_frames/frame_rate` seconds, same pitch, velocity scaled by
    `echo_cfg.volume`, tagged HARMONY, then admitted only into a clear gap between
    comp notes (comp wins). The merged result is strictly monophonic.
    `harmony_notes` may come from either harmony_source (both are monophonic).
    """
    delay = echo_cfg.delay_frames / frame_rate
    comp = sorted(harmony_notes, key=lambda n: n.start)
    echoes: list[NoteEvent] = []
    for n in lead_notes:
        if n.end - n.start <= echo_cfg.min_lead_seconds:
            continue
        vel = max(0, min(127, round(n.velocity * echo_cfg.volume)))
        cand = NoteEvent(pitch=n.pitch, start=n.start + delay, end=n.end + delay,
                         velocity=vel, role=Role.HARMONY)
        clamped = _clamp_to_gap(cand, comp)
        if clamped is not None:
            echoes.append(clamped)
    return _enforce_mono(comp + echoes)
