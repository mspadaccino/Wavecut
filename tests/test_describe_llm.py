"""Il lettore a modello, provato senza rete: un client finto che risponde
col modulo, e le letture tenute su disco."""

import pytest

pytest.importorskip("pydantic", reason="pydantic arriva con `anthropic`")

from core.analysis.describe import Query, Vocabulary
from core.analysis.describe_llm import (ClaudeReader, Readings, ReadingFailed,
                                        vocabulary_prompt)

VOCABULARY = Vocabulary(genres=["Electronic - Synth-pop", "Rock - New Wave"],
                        moods=["happy", "dark"])


class _Response:
    def __init__(self, parsed, stop_reason="end_turn") -> None:
        self.parsed_output = parsed
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class _Client:
    def __init__(self, answer) -> None:
        self.messages = _Messages(answer)


def _form(**fields):
    from core.analysis.describe_llm import _schema
    return _schema()(**fields)


def test_the_form_comes_back_as_a_cleaned_query():
    client = _Client(_Response(_form(
        years=[1980, 1989], genres=["electronic - synth-pop", "Electronic - Trance"],
        moods=["DARK"], title_words=["extended"], how_read="80s synth pop, extended")))
    query = ClaudeReader(client=client).read("synth pop anni 80 extended", VOCABULARY)
    assert query.years == (1980, 1989)
    assert query.genres == ["Electronic - Synth-pop"]     # la grafia della libreria
    assert query.moods == ["dark"]
    assert query.title_words == ["extended"]
    assert query.how_read == "80s synth pop, extended"


def test_what_goes_over_the_wire_is_the_phrase_and_the_vocabulary_only():
    client = _Client(_Response(_form()))
    ClaudeReader(client=client, model="claude-test").read("  90s  ", VOCABULARY)
    sent = client.messages.calls[0]
    assert sent["model"] == "claude-test"
    assert sent["messages"] == [{"role": "user", "content": "90s"}]
    system = sent["system"][0]["text"]
    assert "- Electronic - Synth-pop" in system and "- dark" in system
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["output_config"] == {"effort": "low"}


def test_the_prompt_is_stable_for_the_same_library():
    one = vocabulary_prompt(Vocabulary(genres=["B", "A"], moods=["y", "x"]))
    two = vocabulary_prompt(Vocabulary(genres=["A", "B"], moods=["x", "y"]))
    assert one == two


def test_an_empty_phrase_never_calls_the_model():
    client = _Client(RuntimeError("must not be called"))
    assert ClaudeReader(client=client).read("   ", VOCABULARY).is_empty()
    assert client.messages.calls == []


def test_trouble_becomes_a_reading_failed_with_a_line_for_the_user():
    class AuthenticationError(Exception):
        pass

    with pytest.raises(ReadingFailed, match="key was refused"):
        ClaudeReader(client=_Client(AuthenticationError("401"))).read("80s", VOCABULARY)
    with pytest.raises(ReadingFailed, match="declined"):
        ClaudeReader(client=_Client(_Response(None, "refusal"))).read("80s", VOCABULARY)
    with pytest.raises(ReadingFailed, match="No API key"):
        ClaudeReader(api_key=None).read("80s", VOCABULARY)


def test_readings_remember_a_phrase_whatever_its_case_and_spacing(tmp_path):
    readings = Readings(tmp_path / "readings.json")
    assert readings.get("80s synth pop") is None
    readings.put("80s  Synth Pop", Query(years=(1980, 1989), how_read="ok"))
    assert readings.get("80s synth pop") == Query(years=(1980, 1989), how_read="ok")
    readings.forget("80S SYNTH POP")
    assert readings.get("80s synth pop") is None


def test_a_broken_readings_file_reads_as_empty(tmp_path):
    path = tmp_path / "readings.json"
    path.write_text("{not json")
    assert Readings(path).get("x") is None
    Readings(path).put("x", Query())
    assert Readings(path).get("x") == Query()


# --- la cura ---

def _picks(*pairs):
    from core.analysis.describe_llm import _curation_schema
    schema = _curation_schema()
    return schema(picks=[{"id": i, "why": why} for i, why in pairs])


def _frame():
    import pandas as pd
    return pd.DataFrame({
        "name": ["a.mp3", "b.mp3", "c.mp3"], "title": ["A", "B", ""],
        "artist": ["Art", "", ""], "year": [1983, None, 1990],
        "bpm": [120.0, None, 128.0], "camelot": ["8A", "", "3B"],
        "genres": ["Electronic - House", "", "Rock - New Wave"],
        "moods": ["happy", "", ""],
    }, index=[10, 20, 30])                              # posizioni di libreria


def test_candidate_lines_are_short_and_skip_what_is_unknown():
    from core.analysis.describe_llm import candidate_line
    frame = _frame()
    assert candidate_line(1, frame.loc[10]) == \
        "1. Art - A | 1983 | 120 bpm | 8A | Electronic - House | happy"
    assert candidate_line(3, frame.loc[30]) == "3. c.mp3 | 1990 | 128 bpm | 3B | Rock - New Wave"


def test_the_curator_keeps_claudes_order_drops_bad_ids_and_caps_at_size():
    from core.analysis.describe_llm import ClaudeCurator
    client = _Client(_Response(_picks((3, "the floor filler"), (9, None),
                                      (1, None), (3, None), (2, None))))
    got = ClaudeCurator(client=client).curate(
        "80s", Query(years=(1980, 1989)), _frame(), candidates=[10, 20, 30], size=2)
    assert got.picks == [30, 10]                        # 3 → 30, 9 cade, 1 → 10, poi basta
    assert got.reasons == {30: "the floor filler"}
    sent = client.messages.calls[0]
    assert "The DJ asked: 80s" in sent["messages"][0]["content"]
    assert "1. Art - A" in sent["messages"][0]["content"]
    assert "pick the best 2 tracks" in " ".join(sent["system"].split())


def test_the_curator_fails_loudly_and_skips_the_empty_case():
    from core.analysis.describe_llm import ClaudeCurator

    class RateLimitError(Exception):
        pass

    with pytest.raises(ReadingFailed, match="Too many requests"):
        ClaudeCurator(client=_Client(RateLimitError())).curate(
            "80s", Query(), _frame(), [10], size=1)
    quiet = _Client(RuntimeError("must not be called"))
    assert ClaudeCurator(client=quiet).curate("80s", Query(), _frame(), [], 5).picks == []
    assert quiet.messages.calls == []


def test_the_curator_gets_more_time_than_the_reader():
    from core.analysis.describe_llm import (CURATE_TIMEOUT, TIMEOUT_SECONDS,
                                            ClaudeCurator)
    assert CURATE_TIMEOUT > TIMEOUT_SECONDS
    assert ClaudeCurator(api_key="sk")._timeout == CURATE_TIMEOUT
    assert ClaudeReader(api_key="sk")._timeout == TIMEOUT_SECONDS
