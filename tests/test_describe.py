"""Crate Buddy: il criterio compilato (`Query`), il vocabolario della libreria e
la ricerca che applica il modulo alla mappa."""

import numpy as np
import pandas as pd

from core.analysis.describe import (Match, Query, Vocabulary, as_xml, pool,
                                    search, seeds, spread, summary)


def library() -> pd.DataFrame:
    rows = [
        # name, title, year, bpm, duration, genres, moods
        ("a.mp3", "Enola Gay (Extended)", 1980, 118.0, 390.0,
         "Electronic - Synth-pop; Rock - New Wave", "melancholic"),
        ("b.mp3", "Blue Monday", 1983, 130.0, 445.0,
         "Rock - New Wave; Electronic - Synth-pop", "dark"),
        ("c.mp3", "Holiday (12\" Mix)", 1983, 116.0, 380.0,
         "Funk / Soul - Disco; Pop", "happy"),
        ("d.mp3", "Show Me Love", 1993, 122.0, 300.0,
         "Electronic - House", "happy"),
        ("e.mp3", "Undated One", None, 124.0, 200.0,
         "Electronic - Synth-pop", "happy"),
        ("f.mp3", "Bolero", 1928, np.nan, 900.0, "Classical", "epic"),
    ]
    frame = pd.DataFrame({
        "name": [r[0] for r in rows], "title": [r[1] for r in rows],
        "year": [r[2] for r in rows], "bpm": [r[3] for r in rows],
        "duration": [r[4] for r in rows], "genres": [r[5] for r in rows],
        "moods": [r[6] for r in rows],
    })
    frame["genre_list"] = frame["genres"].str.split("; ")
    frame["mood_list"] = frame["moods"].str.split("; ")
    return frame


def fan(n: int) -> np.ndarray:
    angles = np.radians(np.linspace(0, 150, n))
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)


# --- il modulo ---

def test_a_query_reads_from_a_loose_dictionary_and_writes_itself_back():
    query = Query.from_dict({"years": ["1980", 1989.0], "genres": "Rock - New Wave",
                             "bpm": (140, 120), "title_words": ["remix", "remix"],
                             "min_minutes": "5.5", "how_read": "80s new wave"})
    assert query.years == (1980, 1989)
    assert query.genres == ["Rock - New Wave"]
    assert query.bpm == (120.0, 140.0)                 # rimessi in ordine
    assert query.title_words == ["remix"]
    assert query.min_minutes == 5.5
    assert Query.from_dict(query.to_dict()) == query


def test_what_cannot_be_read_falls_instead_of_breaking():
    query = Query.from_dict({"years": "eighties", "bpm": [None, 3], "min_minutes": -2,
                             "genres": None, "how_read": None})
    assert query.years is None and query.bpm is None
    assert query.min_minutes is None and query.genres == []
    assert query.is_empty()
    assert Query.from_dict("garbage").is_empty()
    assert Query.from_dict({"years": [1700, 1750]}).years is None


def test_cleaning_brings_labels_to_the_library_spelling_and_drops_the_rest():
    vocabulary = Vocabulary.of(library())
    assert vocabulary.genres[0] == "Electronic - Synth-pop"     # il più frequente
    query = Query(genres=["electronic - synth-pop", "Electronic - Trance"],
                  moods=["HAPPY", "gloomy"]).cleaned(vocabulary)
    assert query.genres == ["Electronic - Synth-pop"]
    assert query.moods == ["happy"]


def test_the_summary_is_one_readable_line():
    query = Query(years=(1980, 1989), genres=["Electronic - Synth-pop"],
                  title_words=["extended", "12\""], min_minutes=5.5)
    assert summary(query) == "1980–1989 · Synth-pop · title has extended / 12\" · ≥ 5.5 min"
    assert summary(Query()) == "everything"


def test_the_criterion_in_xml_separates_filters_from_seeds():
    query = Query(years=(1980, 1989), genres=["Electronic - Synth-pop"],
                  moods=["happy"], bpm=(118.4, 130.0),
                  title_words=["extended", "12\""], min_minutes=5.5)
    assert as_xml(query, phrase="synth pop anni 80", read_by="Claude") == (
        '<search phrase="synth pop anni 80" read-by="Claude">\n'
        "  <filters>\n"
        '    <years from="1980" to="1989"/>\n'
        '    <bpm from="118" to="130"/>\n'
        "    <title-words>extended, 12&quot;</title-words>\n"
        '    <min-length minutes="5.5"/>\n'
        "  </filters>\n"
        "  <seeds>\n"
        "    <genres>Electronic - Synth-pop</genres>\n"
        "    <moods>happy</moods>\n"
        "  </seeds>\n"
        "</search>")


def test_a_criterion_that_reads_nothing_says_so_instead_of_lying():
    # Niente sezioni vuote: o c'è un vincolo, o si dice che non ce n'è.
    assert as_xml(Query()) == ("<search>\n"
                               "  <!-- nothing read: every track passes -->\n"
                               "</search>")
    assert as_xml(Query(genres=["Rock - New Wave"])) == (
        "<search>\n  <seeds>\n    <genres>Rock - New Wave</genres>\n"
        "  </seeds>\n</search>")


def test_the_criterion_escapes_what_would_break_the_xml():
    told = as_xml(Query(title_words=["rock & roll", "<12\">"]),
                  phrase='the "best" of rock & roll')
    assert 'phrase="the &quot;best&quot; of rock &amp; roll"' in told
    assert "<title-words>rock &amp; roll, &lt;12&quot;&gt;</title-words>" in told


# --- i filtri duri ---

def test_years_keep_the_dated_and_drop_the_undated():
    assert pool(library(), Query(years=(1980, 1989))) == [0, 1, 2]


def test_a_missing_tempo_passes_a_tempo_filter_but_a_short_track_not_a_length_one():
    frame = library()
    assert pool(frame, Query(bpm=(115, 120))) == [0, 2, 5]      # Bolero non ha BPM
    assert pool(frame, Query(min_minutes=6)) == [0, 1, 2, 5]


def test_title_words_are_alternatives_and_fold_case_and_accents():
    frame = library()
    assert pool(frame, Query(title_words=["EXTENDED", "12\""])) == [0, 2]
    assert pool(frame, Query(title_words=["énola"])) == [0]


def test_a_year_filter_on_a_map_without_years_keeps_nothing():
    frame = library().drop(columns=["year"])
    assert pool(frame, Query(years=(1980, 1989))) == []


# --- i semi ---

def test_seeds_rank_by_how_strongly_the_label_is_carried():
    frame = library()
    query = Query(genres=["Electronic - Synth-pop"])
    # a e e portano Synth-pop per primo; b per secondo.
    assert seeds(frame, query, list(frame.index)) == [0, 4, 1]
    both = Query(genres=["Electronic - Synth-pop"], moods=["dark"])
    assert seeds(frame, both, list(frame.index))[0] == 1       # genere ½ + mood 1


def test_seeds_count_a_song_once_and_need_labels():
    frame = library()
    query = Query(genres=["Electronic - Synth-pop"])
    same = seeds(frame, query, list(frame.index), song_of=lambda i: "one song")
    assert same == [0]
    assert seeds(frame, Query(years=(1980, 1989)), list(frame.index)) == []


def test_spread_covers_the_whole_row():
    assert spread(list(range(10)), 3) == [0, 4, 9]
    assert spread([1, 2], 5) == [1, 2]
    assert spread(list(range(10)), 0) == []


# --- la ricerca ---

def test_search_puts_the_seeds_first_and_fills_from_the_pool():
    frame = library()
    query = Query(years=(1980, 1989), genres=["Electronic - Synth-pop"])
    found = search(frame, fan(len(frame)), query, size=3)
    assert isinstance(found, Match)
    assert found.pool == [0, 1, 2]
    assert found.seeds == [0, 1]
    assert found.tracks[:2] == [0, 1]
    assert found.tracks[2] == 2                     # il riempimento resta nel pool
    assert found.no_year == 1


def test_search_without_labels_samples_the_pool_across_the_years():
    frame = library()
    found = search(frame, fan(len(frame)), Query(min_minutes=5), size=2)
    assert found.seeds == []
    assert found.tracks == [5, 3]                   # dal 1928 agli anni 90
    assert len(found.tracks) == 2


def test_search_stops_at_size_and_survives_no_embeddings():
    frame = library()
    query = Query(genres=["Electronic - Synth-pop"])
    assert len(search(frame, fan(len(frame)), query, size=1).tracks) == 1
    assert search(frame, None, query, size=5).tracks == [0, 4, 1]


# --- l'anno stimato ---

def test_a_confident_guess_dates_a_track_the_tags_did_not():
    from core.analysis.describe import guessed_years, years_of
    frame = library()
    frame["year_guess"] = [None, None, None, None, 1984.0, 1983.0]
    frame["year_guess_conf"] = [0, 0, 0, 0, 0.9, 0.3]
    years = years_of(frame)
    assert years.at[4] == 1984                          # la stima sicura entra
    assert years.at[0] == 1980                          # il tag vince sempre
    assert years.at[5] == 1928                          # il tag c'era
    assert pool(frame, Query(years=(1980, 1989))) == [0, 1, 2, 4]
    assert list(guessed_years(frame)) == [False, False, False, False, True, False]
    found = search(frame, fan(len(frame)), Query(years=(1980, 1989)), size=10)
    assert found.guessed == 1 and found.no_year == 0


def test_a_weak_guess_does_not_date():
    from core.analysis.describe import years_of
    frame = library()
    frame["year_guess"] = [None, None, None, None, 1984.0, None]
    frame["year_guess_conf"] = [0, 0, 0, 0, 0.5, 0]
    assert np.isnan(years_of(frame).at[4])
    assert search(frame, None, Query(years=(1980, 1989)), size=10).no_year == 1
