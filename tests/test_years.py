"""L'anno di un brano: dai tag, dal nome, e a posteriori sulle righe di una
mappa fatta prima che si leggesse."""

from pathlib import Path

import numpy as np

from core.analysis import years
from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore
from tests.test_titles import _FRAME, tagged_mp3


def dated_mp3(path: Path, date: str = "", original: str = "") -> Path:
    from mutagen.id3 import ID3, TDOR, TDRC

    path.write_bytes(_FRAME * 4)
    if date or original:
        tags = ID3()
        if date:
            tags.add(TDRC(encoding=3, text=date))
        if original:
            tags.add(TDOR(encoding=3, text=original))
        tags.save(path)
    return path


def test_the_original_date_wins_over_the_release_date(tmp_path):
    reissue = dated_mp3(tmp_path / "a.mp3", date="2011-05-02", original="1983")
    assert years.year_from_tags(reissue) == 1983
    assert years.year_from_tags(dated_mp3(tmp_path / "b.mp3", date="1985")) == 1985


def test_no_tag_or_no_file_is_no_year(tmp_path):
    assert years.year_from_tags(tagged_mp3(tmp_path / "bare.mp3")) is None
    assert years.year_from_tags(tmp_path / "nope.mp3") is None
    assert years.year_from_tags(dated_mp3(tmp_path / "odd.mp3", date="n/a")) is None


def test_the_name_gives_a_year_only_in_brackets_or_leading_a_folder():
    assert years.year_from_name("/x/Madonna - Holiday (1983).mp3") == 1983
    assert years.year_from_name("/x/Thriller [1982]/Beat It.mp3") == 1982
    assert years.year_from_name("/x/1979 - Off The Wall/Rock With You.mp3") == 1979
    # Un quattro cifre sciolto è un titolo, non una data.
    assert years.year_from_name("/x/Pulp - Disco 2000.mp3") is None
    assert years.year_from_name("/x/Prince - 1999.mp3") is None
    # Fra parentesi ma non un anno: un numero di catalogo.
    assert years.year_from_name("/x/Track (0042).mp3") is None


def test_year_of_reads_the_tags_first_and_the_name_after(tmp_path):
    folder = tmp_path / "Album (1990)"
    folder.mkdir()
    tagged = dated_mp3(folder / "one.mp3", date="1987")
    untagged = tagged_mp3(folder / "two.mp3")
    assert years.year_of(tagged) == 1987
    assert years.year_of(untagged) == 1990


def test_the_profile_row_carries_the_year():
    assert TrackProfile(path=Path("/x/a.mp3"), year=1983).to_row()["year"] == 1983
    assert TrackProfile(path=Path("/x/a.mp3")).to_row()["year"] is None


def _profile(path, vector):
    return TrackProfile(path=path, duration=300.0, bpm=128.0, camelot="8A",
                        embedding=np.full(EMBEDDING_DIM, vector, dtype=np.float32))


def test_the_backfill_visits_only_the_rows_without_the_field(tmp_path):
    old = dated_mp3(tmp_path / "old.mp3", date="1984")
    bare = tagged_mp3(tmp_path / "bare.mp3")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(old, 1.0), _profile(bare, 2.0)])
    for row in store.rows:
        del row["year"]                     # una mappa di prima
    store.rewrite()

    assert years.missing(store.rows) == [0, 1]
    assert years.backfill(store) == 2
    again = MapStore.load(tmp_path / "map")
    assert again.rows[0]["year"] == 1984
    assert again.rows[1]["year"] is None    # visitato, senza anno
    assert years.missing(again.rows) == []
    assert years.known(again.rows) == 1


def test_a_missing_file_is_left_for_next_time(tmp_path):
    gone = tmp_path / "gone.mp3"
    dated_mp3(gone, date="1984")
    store = MapStore.load(tmp_path / "map")
    store.append([_profile(gone, 1.0)])
    del store.rows[0]["year"]
    gone.unlink()
    assert years.backfill(store) == 0
    assert years.missing(store.rows) == [0]
