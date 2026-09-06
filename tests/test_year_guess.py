"""L'anno stimato da Claude via batch: chi si chiede, come, e come le
risposte tornano sulle righe — con un'API finta."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.analysis import year_guess
from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore


def _rows():
    return [
        {"path": "/x/New Order - Blue Monday.mp3", "title": "Blue Monday",
         "artist": "New Order", "year": None},
        {"path": "/x/Thriller/Beat It.mp3", "title": "", "artist": "",
         "year": None},
        {"path": "/x/dated.mp3", "title": "Dated", "artist": "", "year": 1990},
        {"path": "/x/asked.mp3", "title": "Asked", "artist": "", "year": None,
         "year_guess": None, "year_guess_conf": 0.0},
    ]


def test_candidates_are_the_undated_never_asked():
    assert year_guess.candidates(_rows()) == [0, 1]


def test_a_line_says_what_is_known_and_nothing_else():
    rows = _rows()
    assert year_guess.line(1, rows[0]) == (
        "1. | New Order - Blue Monday | file: New Order - Blue Monday | folder: x")
    assert year_guess.line(2, rows[1]) == "2. | file: Beat It | folder: Thriller"


def test_a_request_packs_the_lines_under_the_schema():
    rows = _rows()
    got = year_guess.request("years-00000", rows, [0, 1], model="claude-test")
    assert got["custom_id"] == "years-00000"
    params = got["params"]
    assert params["model"] == "claude-test"
    assert params["output_config"]["format"]["schema"] == year_guess.ANSWER_SCHEMA
    assert params["messages"][0]["content"].splitlines()[0].startswith("1. |")
    assert params["messages"][0]["content"].splitlines()[1].startswith("2. |")
    assert year_guess.chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_answers_are_read_with_tolerance():
    text = json.dumps({"tracks": [
        {"id": 1, "year": 1983, "confidence": 0.95},
        {"id": 2, "year": None, "confidence": 0},
        {"id": 3, "year": 1700, "confidence": 1.4},
        {"id": "x", "year": 1980, "confidence": 0.5},
    ]})
    assert year_guess.parse_answer(text) == {
        1: (1983, 0.95), 2: (None, 0.0), 3: (None, 1.0)}
    assert year_guess.parse_answer("not json") == {}
    assert year_guess.parse_answer('noise {"tracks": [{"id": 1, "year": 1990, '
                                   '"confidence": 0.7}]} more') == {1: (1990, 0.7)}


def test_apply_writes_the_guess_beside_the_tag_and_leaves_the_unanswered():
    rows = _rows()
    dated = year_guess.apply(rows, {1: 0, 2: 1}, {1: (1983, 0.95)})
    assert dated == 1
    assert rows[0]["year_guess"] == 1983 and rows[0]["year_guess_conf"] == 0.95
    assert rows[0]["year"] is None                      # il tag non si tocca
    assert "year_guess" not in rows[1]                  # resta da chiedere


# --- con un'API finta ---

class _Batches:
    def __init__(self) -> None:
        self.sent = []
        self.status = "in_progress"
        self.answers = {}

    def create(self, requests):
        self.sent = requests
        return SimpleNamespace(id="msgbatch_test")

    def retrieve(self, batch_id):
        return SimpleNamespace(
            processing_status=self.status,
            request_counts=SimpleNamespace(processing=len(self.sent),
                                           succeeded=0, errored=0))

    def results(self, batch_id):
        for custom_id, text in self.answers.items():
            if text is None:
                yield SimpleNamespace(custom_id=custom_id,
                                      result=SimpleNamespace(type="errored"))
                continue
            yield SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=text)])))


class _Client:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(batches=_Batches())


def _store(tmp_path) -> MapStore:
    tmp_path.mkdir(exist_ok=True)
    store = MapStore.load(tmp_path / "map")
    profiles = []
    for k, name in enumerate(("a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3")):
        audio = tmp_path / name
        audio.write_bytes(b"x")
        profiles.append(TrackProfile(
            path=audio, duration=300.0, bpm=120.0, camelot="8A",
            year=1990 if k == 4 else None,
            embedding=np.full(EMBEDDING_DIM, float(k), dtype=np.float32)))
    store.append(profiles)
    return store


def test_submit_sends_the_undated_in_chunks_and_remembers_the_lot(tmp_path):
    store, client = _store(tmp_path), _Client()
    lot = year_guess.submit(client, store, model="claude-test", per_request=3)
    assert lot.batch_id == "msgbatch_test" and lot.model == "claude-test"
    assert [len(p) for p in lot.requests.values()] == [3, 1]        # 4 senza anno
    assert len(client.messages.batches.sent) == 2
    year_guess.save_lot(store.directory, lot)
    again = year_guess.pending_lots(store.directory)
    assert len(again) == 1 and again[0].requests == lot.requests
    assert year_guess.submit(client, store, limit=0) is not None      # non ancora chiesti
    assert year_guess.submit(_Client(), _store(tmp_path / "other"), limit=2).tracks == 2


def test_collect_waits_then_writes_by_path_and_forgets_the_lot(tmp_path):
    store, client = _store(tmp_path), _Client()
    lot = year_guess.submit(client, store, per_request=3)
    year_guess.save_lot(store.directory, lot)

    waiting = year_guess.collect(client, store, lot)
    assert not waiting.ended and waiting.processing == 2
    assert "year_guess" not in store.rows[0]

    batches = client.messages.batches
    batches.status = "ended"
    batches.answers = {
        "years-00000": json.dumps({"tracks": [
            {"id": 1, "year": 1983, "confidence": 0.9},
            {"id": 2, "year": None, "confidence": 0.0},
            {"id": 3, "year": 1991, "confidence": 0.4}]}),
        "years-00001": None,                            # richiesta fallita
    }
    got = year_guess.collect(client, store, lot)
    assert got.ended and got.answered == 1 and got.dated == 2 and got.failed == 1
    again = MapStore.load(store.directory)
    assert again.rows[0]["year_guess"] == 1983
    assert again.rows[1]["year_guess"] is None
    assert again.rows[2]["year_guess"] == 1991 and again.rows[2]["year_guess_conf"] == 0.4
    assert "year_guess" not in again.rows[3]            # la richiesta fallita resta da chiedere
    assert year_guess.candidates(again.rows) == [3]
    year_guess.forget_lot(store.directory, lot)
    assert year_guess.pending_lots(store.directory) == []


def test_a_track_gone_from_the_map_loses_its_guess_without_shifting_the_others(tmp_path):
    store, client = _store(tmp_path), _Client()
    lot = year_guess.submit(client, store, per_request=3)
    store.remove([store.rows[1]["path"]])               # via b.mp3, il numero 2
    batches = client.messages.batches
    batches.status = "ended"
    batches.answers = {"years-00000": json.dumps({"tracks": [
        {"id": 1, "year": 1983, "confidence": 0.9},
        {"id": 2, "year": 2001, "confidence": 0.9},
        {"id": 3, "year": 1991, "confidence": 0.9}]})}
    year_guess.collect(client, store, lot)
    by_name = {Path(r["path"]).name: r for r in store.rows}
    assert by_name["a.mp3"]["year_guess"] == 1983
    assert by_name["c.mp3"]["year_guess"] == 1991       # il 3 resta il 3
