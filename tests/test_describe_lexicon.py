"""Il lettore a regole: epoche, etichette per nome, parole del mestiere."""

from core.analysis.describe import Query, Vocabulary
from core.analysis.describe_lexicon import genres_in, moods_in, read, years_in

VOCABULARY = Vocabulary(
    genres=["Electronic - House", "Electronic - Synth-pop", "Rock - New Wave",
            "Funk / Soul - Disco", "Electronic - Italo-Disco",
            "Electronic - Eurodance", "Electronic - Deep House",
            "Rock - Pop Rock", "Rock - Classic Rock", "Pop - Ballad",
            "Electronic - Hi NRG", "Electronic - Freestyle",
            "Hip Hop - Boom Bap", "Latin - Salsa", "Electronic - Italo House"],
    moods=["happy", "dark", "melancholic", "romantic", "party", "summer",
           "energetic"])


# --- le epoche ---

def test_decades_in_every_way_they_are_written():
    for text in ("70s", "'70s", "anni 70", "anni '70", "1970s", "seventies",
                 "anni settanta", "the 70's"):
        assert years_in(text) == (1970, 1979), text
    assert years_in("00s") == (2000, 2009)
    assert years_in("anni 2000") == (2000, 2009)
    assert years_in("anni 1980") == (1980, 1989)


def test_two_decades_or_two_years_fuse_into_one_span():
    assert years_in("70s e 80s") == (1970, 1989)
    assert years_in("1985-1992") == (1985, 1992)
    assert years_in("dal 1985 al 1992") == (1985, 1992)
    assert years_in("1983") == (1983, 1983)


def test_open_ended_years():
    assert years_in("dal 1985") == (1985, 2100)
    assert years_in("before 1990") == (1900, 1990)


def test_a_bare_number_is_not_a_decade():
    assert years_in("80 bpm") is None
    assert years_in("120-128 bpm") is None
    assert years_in("disco") is None


# --- le etichette ---

def test_a_leaf_is_matched_whole_however_it_is_spelled():
    for text in ("synth pop", "synthpop", "Synth-Pop", "synth-pop anni 80"):
        assert genres_in(text, VOCABULARY) == ["Electronic - Synth-pop"], text
    assert genres_in("new wave", VOCABULARY) == ["Rock - New Wave"]
    assert genres_in("house", VOCABULARY) == ["Electronic - House"]     # non Deep House


def test_a_macro_genre_word_takes_all_its_leaves():
    assert genres_in("rock", VOCABULARY) == [
        "Rock - New Wave", "Rock - Pop Rock", "Rock - Classic Rock"]
    assert genres_in("funk", VOCABULARY) == ["Funk / Soul - Disco"]
    assert genres_in("elettronica", VOCABULARY)[0] == "Electronic - House"
    assert genres_in("rap", VOCABULARY) == ["Hip Hop - Boom Bap"]


def test_moods_by_name_and_in_italian():
    assert moods_in("dark and happy", VOCABULARY) == ["happy", "dark"]
    assert moods_in("brani allegri e romantici", VOCABULARY) == []
    assert moods_in("cupo", VOCABULARY) == ["dark"]


# --- le parole del mestiere ---

def test_the_nine_collections_of_the_shop():
    read_ = lambda text: read(text, VOCABULARY)   # noqa: E731
    assert read_("70s").years == (1970, 1979)
    assert read_("80s").years == (1980, 1989)
    assert read_("90s").years == (1990, 1999)
    flash = read_("Flash House")
    assert flash.years == (1986, 1993)
    assert "Electronic - House" in flash.genres
    assert "Electronic - Hi NRG" in flash.genres
    ballads = read_("Ballads Remixes")
    assert ballads.bpm == (40.0, 105.0)
    assert ballads.title_words == ["remix"]
    assert ballads.moods == ["melancholic", "romantic"] or \
        set(ballads.moods) == {"romantic", "melancholic"}
    assert "Pop - Ballad" in ballads.genres
    wave = read_("New Wave / Synth Pop")
    assert set(wave.genres) == {"Rock - New Wave", "Electronic - Synth-pop"}
    assert read_("ReVibes").title_words[:2] == ["rework", "re-edit"]
    assert read_("Rock").genres == genres_in("rock", VOCABULARY)
    assert read_("Italo House").genres == ["Electronic - Italo House"]
    assert set(read_("italo").genres) == {"Electronic - Italo-Disco",
                                          "Electronic - Italo House"}
    assert read_("Eurodance").genres == ["Electronic - Eurodance"]


def test_a_phrase_in_italian_and_in_english_read_the_same():
    it = read("synth pop anni 80, solo versioni extended", VOCABULARY)
    en = read("80s synth pop, extended versions only", VOCABULARY)
    assert it.years == en.years == (1980, 1989)
    assert it.genres == en.genres == ["Electronic - Synth-pop"]
    assert it.title_words == en.title_words == ["extended"]
    assert it.how_read == "1980–1989 · Synth-pop · title has extended"


def test_tempo_and_length_words():
    assert read("120-128 bpm", VOCABULARY).bpm == (120.0, 128.0)
    assert read("sotto i 100 bpm", VOCABULARY).bpm == (40.0, 100.0)
    assert read("over 125 bpm", VOCABULARY).bpm == (125.0, 300.0)
    assert read("almeno 6 minuti", VOCABULARY).min_minutes == 6.0
    assert read("versioni lunghe", VOCABULARY).min_minutes == 5.5
    assert read("12 inch", VOCABULARY).title_words[0] == "12\""


def test_a_phrase_the_lexicon_does_not_know_reads_as_everything():
    query = read("something quite else", VOCABULARY)
    assert isinstance(query, Query) and query.is_empty()
    assert query.how_read == "everything"
