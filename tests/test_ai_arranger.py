import pytest
from chiptune.score import NoteEvent, Role, Score, TempoGrid
from chiptune.config import load_config
from chiptune.arrange import ai_arranger

def _score():
    return Score(TempoGrid(120.,0.,4),
                 [NoteEvent(72,0.,0.5,100,Role.LEAD), NoteEvent(74,0.5,1.0,100,Role.LEAD)], 1.0)

def _heuristic():
    return Score(TempoGrid(120.,0.,4), [NoteEvent(60,0.,1.0,100,Role.LEAD)], 1.0)

def test_arrange_parses_llm_output_into_a_score(monkeypatch):
    monkeypatch.setattr(ai_arranger, "_call_llm",
        lambda prompt, cfg: "KEY: C maj\nLEAD: 1:1 2:1\nBASS: 1:2\nDRUMS: K:1 S:1")
    cfg = load_config()
    out = ai_arranger.arrange(_score(), cfg.ai, {"LEAD":4,"HARM":3,"BASS":2}, _heuristic)
    roles = {n.role for n in out.notes}
    assert Role.LEAD in roles and Role.BASS in roles and Role.PERCUSSION in roles
    assert out is not _heuristic()  # used the LLM, not the fallback

def test_arrange_falls_back_on_llm_error(monkeypatch):
    def boom(prompt, cfg): raise RuntimeError("api down")
    monkeypatch.setattr(ai_arranger, "_call_llm", boom)
    cfg = load_config()
    out = ai_arranger.arrange(_score(), cfg.ai, {"LEAD":4,"HARM":3,"BASS":2}, _heuristic)
    assert [n.pitch for n in out.notes] == [60]  # the heuristic fallback score

def test_arrange_falls_back_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(ai_arranger, "_call_llm", lambda prompt, cfg: "sorry i cannot help")
    cfg = load_config()
    out = ai_arranger.arrange(_score(), cfg.ai, {"LEAD":4,"HARM":3,"BASS":2}, _heuristic)
    assert [n.pitch for n in out.notes] == [60]

def test_format_prompt_includes_the_melody(monkeypatch):
    p = ai_arranger.format_prompt(_score())
    assert "MELODY" in p.upper() and "120" in p  # tempo present
