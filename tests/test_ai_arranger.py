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
    # The LLM path was taken, not the fallback: "LEAD: 2" is D4 (62) in C major
    # oct 4, a pitch the heuristic fallback (its only note is 60) never emits.
    assert 62 in {n.pitch for n in out.notes}

def test_arrange_caps_runaway_length(monkeypatch):
    # A teacher that writes a bass line far longer than the song must be truncated
    # to ~the song length (1.5x), not run out to the 600s buffer rail.
    runaway = "KEY: C maj\nLEAD: 1:1\nBASS: " + " ".join(["1:1"] * 400)
    monkeypatch.setattr(ai_arranger, "_call_llm", lambda p, c: runaway)
    cfg = load_config()
    src = _score()  # 1.0 s long
    out = ai_arranger.arrange(src, cfg.ai, {"LEAD":4,"HARM":3,"BASS":2}, _heuristic)
    cap = max(src.duration * 1.5, src.duration + 8.0)
    assert out.duration <= cap + 1e-6                      # capped near the song, not 200s of bass
    assert out.duration < 20.0                             # far below the uncapped ~200s runaway
    assert any(n.role is Role.BASS for n in out.notes)     # some bass survived (AI path, not fallback)

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

def test_format_prompt_includes_the_melody():
    p = ai_arranger.format_prompt(_score())
    assert "MELODY" in p.upper() and "120" in p  # tempo present

def test_format_prompt_includes_chords_and_bass():
    # #1 informed AI: the detected chord progression + real bass line reach the LLM,
    # so it arranges the actual harmony instead of guessing from the melody.
    from chiptune.analysis.chords import ChordSegment
    sc = Score(TempoGrid(120., 0., 4),
               [NoteEvent(72, 0., 0.5, 100, Role.LEAD), NoteEvent(36, 0., 1.0, 100, Role.BASS)], 1.0)
    p = ai_arranger.format_prompt(sc, [ChordSegment(0.0, 2.0, 0, False), ChordSegment(2.0, 4.0, 9, True)])
    assert "CHORDS" in p and "C:" in p and "Am:" in p   # C major -> A minor progression present
    assert "BASS" in p                                    # the real bass line is included
