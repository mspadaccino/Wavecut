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
