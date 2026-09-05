"""Titolo e artista dai tag: letti al profilo, e a posteriori sulle righe di
una mappa fatta prima che si leggessero."""

from pathlib import Path

import numpy as np

from core.analysis import titles
from core.analysis.map_profile import (EMBEDDING_DIM, TrackProfile,
                                       read_tag_title_artist)
from core.analysis.map_store import MapStore

# Un frame MPEG1 layer III a 128 kbps e 44,1 kHz e' lungo 417 byte: quattro
# in fila bastano a mutagen per riconoscere un mp3 e leggerne i tag.
_FRAME = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 413


def tagged_mp3(path: Path, title: str = "", artist: str = "") -> Path:
    """Un mp3 minimo, con i tag chiesti (e senza tag se non se ne chiede)."""
    from mutagen.id3 import ID3, TIT2, TPE1

    path.write_bytes(_FRAME * 4)
    if title or artist:
        tags = ID3()
        if title:
            tags.add(TIT2(encoding=3, text=title))
        if artist:
            tags.add(TPE1(encoding=3, text=artist))
        tags.save(path)
    return path


def _profile(path, vector):
    return TrackProfile(path=path, duration=300.0, bpm=128.0, camelot="8A",
                        embedding=np.full(EMBEDDING_DIM, vector, dtype=np.float32))


def test_the_tags_are_read_and_their_absence_is_two_empty_strings(tmp_path):
    home = tagged_mp3(tmp_path / "a.mp3", "Home", "Julie McKnight")
    assert read_tag_title_artist(home) == ("Home", "Julie McKnight")
    assert read_tag_title_artist(tagged_mp3(tmp_path / "b.mp3")) == ("", "")
    assert read_tag_title_artist(tmp_path / "nope.mp3") == ("", "")


def test_the_row_carries_the_title_and_the_artist_next_to_the_file_name():
    row = TrackProfile(path=Path("/x/Track 08.mp3"), title="Home",
                       artist="Julie McKnight").to_row()
    assert row["name"] == "Track 08.mp3"
    assert row["title"] == "Home"
    assert row["artist"] == "Julie McKnight"


def test_a_profile_without_tags_writes_them_empty_not_missing():
    row = TrackProfile(path=Path("/x/a.mp3")).to_row()
    assert row["title"] == "" and row["artist"] == ""


def test_the_backfill_reads_only_the_rows_the_tags_never_visited(tmp_path):
    old = tagged_mp3(tmp_path / "old.mp3", "Home", "Julie McKnight")
    bare = tagged_mp3(tmp_path / "bare.mp3")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(old, 1.0), _profile(bare, 2.0)])
    # Una mappa di prima: le righe non hanno il campo.
    for row in store.rows:
        del row["title"], row["artist"]
    store.rewrite()

    assert titles.missing(store.rows) == [0, 1]
    assert titles.backfill(store, flush_every=1) == 2
    assert store.rows[0]["title"] == "Home"
    assert store.rows[0]["artist"] == "Julie McKnight"
    # Senza tag si scrive il vuoto, e il vuoto e' "gia' visto".
    assert store.rows[1]["title"] == "" and store.rows[1]["artist"] == ""
    assert titles.missing(store.rows) == []

    again = MapStore.load(tmp_path / "map")
    assert again.rows[0]["title"] == "Home"
    assert again.embeddings.shape == (2, EMBEDDING_DIM)


def test_a_file_that_is_not_reachable_is_left_for_next_time(tmp_path):
    gone = tagged_mp3(tmp_path / "gone.mp3", "Home", "Julie McKnight")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(gone, 1.0)])
    del store.rows[0]["title"], store.rows[0]["artist"]
    gone.unlink()

    assert titles.backfill(store) == 0
    assert titles.missing(store.rows) == [0]
