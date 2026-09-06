"""La scheda Describe: la frase letta nel modulo, il modulo corretto a mano,
la ricerca e la playlist sullo scaffale. Gira solo col gruppo `qt`."""

import os

import pytest

pytest.importorskip("PySide6", reason="gruppo poetry `qt` non installato")
pytest.importorskip("pytestqt", reason="gruppo poetry `qt` non installato")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings

from core.analysis.describe import Query
from core.analysis.describe_llm import Readings, ReadingFailed
from core.analysis.mixing import TransitionCost


def library() -> pd.DataFrame:
    rows = [
        ("one.mp3", "Enola Gay (Extended)", 1980, 118.0, "8A",
         "Electronic - Synth-pop; Rock - New Wave", "melancholic", 390.0),
        ("two.mp3", "Blue Monday", 1983, 130.0, "3B",
         "Rock - New Wave; Electronic - Synth-pop", "dark", 445.0),
        ("three.mp3", "Show Me Love", 1993, 122.0, "8B",
         "Electronic - House", "happy", 300.0),
        ("four.mp3", "Undated", None, 124.0, "5A",
         "Electronic - Synth-pop", "happy", 200.0),
    ]
    frame = pd.DataFrame({
        "name": [r[0] for r in rows], "title": [r[1] for r in rows],
        "artist": "", "year": [r[2] for r in rows],
        "bpm": [r[3] for r in rows], "camelot": [r[4] for r in rows],
        "genres": [r[5] for r in rows], "moods": [r[6] for r in rows],
        "duration": [r[7] for r in rows],
        "energy": [0.2, 0.5, 0.8, 0.6], "danceability": [0.5, 0.6, 0.7, 0.5],
        "valence_rank": [0.3, 0.1, 0.9, 0.8], "folder": "/x",
    })
    frame["path"] = "/x/" + frame["name"]
    frame["top_genre"] = frame["genres"].str.split("; ").str[0]
    frame["genre_list"] = frame["genres"].str.split("; ")
    frame["mood_list"] = frame["moods"].str.split("; ")
    return frame


class _Store:
    def __init__(self, n: int) -> None:
        angles = np.radians(np.linspace(0, 120, n))
        self.embeddings = np.column_stack(
            [np.cos(angles), np.sin(angles)]).astype(np.float32)


class _Keys:
    def __init__(self, key=None) -> None:
        self.key = key

    def read(self):
        return self.key

    def write(self, key):
        self.key = key.strip() or None
        return "a test keyring"

    def where(self):
        return "a test keyring"

    def forget(self):
        self.key = None


class _Reader:
    """Il lettore a modello finto: risponde con la Query data, o fallisce."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls = 0

    def read(self, text, vocabulary):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _panel(qtbot, tmp_path, monkeypatch, keys=None, reader=None):
    # Il lavoro nel pool si fa subito, sul filo: il test non aspetta.
    monkeypatch.setattr("qt_app.pages.map.describe_panel.run_in_pool",
                        lambda job, done, failed=None: _run_now(job, done, failed))
    from core.analysis.shelf import Shelf
    from qt_app.pages.map.describe_panel import DescribePanel
    from qt_app.pages.map.library import Library

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_Store(len(frame)), frame=frame, common={},
                  at_path=at_path,
                  cost=TransitionCost(_Store(len(frame)).embeddings,
                                      frame["bpm"].tolist(),
                                      frame["camelot"].tolist()))
    panel = DescribePanel(
        wire_table=lambda table: None, shelf=Shelf(tmp_path / "shelf"),
        readings=Readings(tmp_path / "readings.json"),
        reader_factory=(lambda key: reader) if reader else None,
        keys=keys or _Keys(),
        settings=QSettings(str(tmp_path / "settings.ini"),
                           QSettings.Format.IniFormat))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    return panel


def _run_now(job, done, failed):
    try:
        result = job()
    except Exception as trouble:                        # noqa: BLE001
        failed(trouble)
        return
    done(result)


def test_a_collection_is_read_by_the_rules_into_the_form(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    panel._collections.setCurrentText("80s")
    assert panel._phrase.text() == "80s"
    assert panel._years_on.isChecked()
    assert (panel._year_from.value(), panel._year_to.value()) == (1980, 1989)
    assert "Read by the rules" in panel._how_read.text()
    assert panel.query().years == (1980, 1989)


def test_without_a_key_the_phrase_is_read_by_the_rules(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    panel._phrase.setText("synth pop anni 80, solo versioni extended")
    panel._on_read()
    query = panel.query()
    assert query.years == (1980, 1989)
    assert query.genres == ["Electronic - Synth-pop"]
    assert query.title_words == ["extended"]
    assert "Add your API key" in panel._reader_told.text()


def test_with_a_key_claude_reads_and_the_reading_is_remembered(qtbot, tmp_path, monkeypatch):
    reader = _Reader(Query(years=(1990, 1999), genres=["Electronic - House"],
                           how_read="house anni 90"))
    panel = _panel(qtbot, tmp_path, monkeypatch, keys=_Keys("sk-test"),
                   reader=reader)
    panel._phrase.setText("house anni 90")
    panel._on_read()
    assert reader.calls == 1
    assert "Read by Claude" in panel._how_read.text()
    assert panel.query().genres == ["Electronic - House"]
    assert panel.query().years == (1990, 1999)

    panel._on_read()                                    # la stessa frase
    assert reader.calls == 1                            # non si ripaga
    assert "Read by memory" in panel._how_read.text()


def test_claude_can_be_switched_off_with_the_key_still_there(qtbot, tmp_path, monkeypatch):
    reader = _Reader(Query(years=(1990, 1999), how_read="from the model"))
    panel = _panel(qtbot, tmp_path, monkeypatch, keys=_Keys("sk-test"),
                   reader=reader)
    panel._ask_claude.setChecked(False)
    assert "nothing is spent" in panel._reader_told.text()
    panel._phrase.setText("90s")
    panel._on_read()
    assert reader.calls == 0                            # niente credito speso
    assert "Read by the rules" in panel._how_read.text()
    assert panel.query().years == (1990, 1999)
    # La scelta si ricorda: un pannello nuovo sulle stesse impostazioni
    # nasce spento.
    again = _panel(qtbot, tmp_path, monkeypatch, keys=_Keys("sk-test"),
                   reader=reader)
    assert not again._ask_claude.isChecked()


def test_when_the_model_fails_the_rules_take_over_and_say_so(qtbot, tmp_path, monkeypatch):
    reader = _Reader(ReadingFailed("The API key was refused: check it under 🔑."))
    panel = _panel(qtbot, tmp_path, monkeypatch, keys=_Keys("sk-bad"),
                   reader=reader)
    panel._phrase.setText("80s")
    panel._on_read()
    assert "key was refused" in panel._reader_told.text()
    assert "Read by the rules" in panel._how_read.text()
    assert panel.query().years == (1980, 1989)
    assert panel._read.isEnabled()


def test_the_form_is_the_query_and_can_be_corrected_by_hand(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    panel._phrase.setText("80s")
    panel._on_read()
    panel._year_to.setValue(1985)                       # corretto a mano
    panel._title_words.setText("extended, 12\"")
    panel._minutes_on.setChecked(True)
    panel._minutes.setValue(6.0)
    query = panel.query()
    assert query.years == (1980, 1985)
    assert query.title_words == ["extended", "12\""]
    assert query.min_minutes == 6.0
    assert query.how_read == "1980–1985 · title has extended / 12\" · ≥ 6 min"


def test_search_lists_the_matches_and_saves_them_under_the_phrase(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    panel._phrase.setText("synth pop anni 80")
    panel._on_read()
    panel._on_search()
    assert sorted(panel._table.paths()) == ["/x/one.mp3", "/x/two.mp3"]
    assert "2 track(s)" in panel._found_told.text()
    assert "1 without a year left out" in panel._found_told.text()

    heard = []
    panel.shelve_playlist.connect(lambda name, indices: heard.append((name, indices)))
    panel._on_shelve()
    assert heard and heard[0][0] == "synth pop anni 80"
    assert sorted(heard[0][1]) == [0, 1]

    added = []
    panel.append_playlist.connect(added.append)
    panel._table.toggle_pick(0)                         # via il primo
    panel._on_add()
    assert len(added) == 1 and len(added[0]) == 1


def test_a_search_that_finds_nothing_says_why(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    panel._years_on.setChecked(True)
    panel._year_from.setValue(1950)
    panel._year_to.setValue(1959)
    panel._on_search()
    assert panel._found_told.text().startswith("No track matches")
    assert not panel._shelve.isEnabled()


def test_playlist_names_come_from_the_phrase():
    from qt_app.pages.map.describe_panel import playlist_name
    assert playlist_name("  synth pop   anni 80 ") == "synth pop anni 80"
    assert playlist_name("a/b") == "Describe"
    assert playlist_name("") == "Describe"


# --- la cura ---

class _Curator:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls = []

    def curate(self, phrase, query, frame, candidates, size):
        self.calls.append((phrase, list(candidates), size))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _curating_panel(qtbot, tmp_path, monkeypatch, curator, key="sk-test"):
    monkeypatch.setattr("qt_app.pages.map.describe_panel.run_in_pool",
                        lambda job, done, failed=None: _run_now(job, done, failed))
    from core.analysis.shelf import Shelf
    from qt_app.pages.map.describe_panel import DescribePanel
    from qt_app.pages.map.library import Library

    frame = library()
    at_path = {frame.at[i, "path"]: i for i in range(len(frame))}
    lib = Library(store=_Store(len(frame)), frame=frame, common={},
                  at_path=at_path,
                  cost=TransitionCost(_Store(len(frame)).embeddings,
                                      frame["bpm"].tolist(),
                                      frame["camelot"].tolist()))
    panel = DescribePanel(
        wire_table=lambda table: None, shelf=Shelf(tmp_path / "shelf"),
        readings=Readings(tmp_path / "readings.json"),
        reader_factory=lambda k: _Reader(Query()),
        curator_factory=lambda k: curator, keys=_Keys(key),
        settings=QSettings(str(tmp_path / "settings.ini"),
                           QSettings.Format.IniFormat))
    qtbot.addWidget(panel)
    panel.set_library(lib)
    return panel


def test_curation_asks_claude_over_a_wider_shortlist_and_keeps_its_picks(qtbot, tmp_path, monkeypatch):
    from core.analysis.describe_llm import Curation
    curator = _Curator(Curation(picks=[1], reasons={1: "the classic"}))
    panel = _curating_panel(qtbot, tmp_path, monkeypatch, curator)
    panel._curate.setChecked(True)
    panel._size.setValue(1)
    panel._phrase.setText("synth pop")
    panel._genres.set_checked(["Electronic - Synth-pop"])
    panel._on_search()
    phrase, candidates, size = curator.calls[0]
    assert phrase == "synth pop" and size == 1
    assert len(candidates) == 3                          # tre per brano voluto
    assert panel._table.paths() == ["/x/two.mp3"]
    assert "Claude kept 1 of 3" in panel._found_told.text()
    assert "the classic" in panel._reasons.text() and panel._reasons.isVisibleTo(panel)
    assert panel._search.isEnabled()


def test_curation_is_off_without_ask_claude_and_falls_back_on_trouble(qtbot, tmp_path, monkeypatch):
    curator = _Curator(ReadingFailed("The key has no credit left."))
    panel = _curating_panel(qtbot, tmp_path, monkeypatch, curator)
    panel._curate.setChecked(True)
    panel._ask_claude.setChecked(False)
    panel._genres.set_checked(["Electronic - Synth-pop"])
    panel._on_search()
    assert curator.calls == []                           # spento: niente rosa
    assert len(panel._table.paths()) == 4                # la lista locale intera

    panel._ask_claude.setChecked(True)
    panel._size.setValue(2)
    panel._on_search()
    assert len(curator.calls) == 1
    assert "no credit" in panel._found_told.text()
    assert len(panel._table.paths()) == 2                # la lista locale, al numero chiesto
    assert not panel._reasons.isVisibleTo(panel)


def test_the_years_hint_counts_claudes_estimates(qtbot, tmp_path, monkeypatch):
    panel = _panel(qtbot, tmp_path, monkeypatch)
    frame = panel._lib.frame
    frame["year_guess"] = [None, None, None, 1987.0]
    frame["year_guess_conf"] = [0, 0, 0, 0.9]
    panel.set_library(panel._lib)
    assert "4 of 4 tracks carry a year (1 estimated by Claude)" in panel._years_hint.text()
    panel._years_on.setChecked(True)
    panel._year_from.setValue(1980)
    panel._year_to.setValue(1989)
    panel._on_search()
    assert "1 dated by Claude's estimate" in panel._found_told.text()
