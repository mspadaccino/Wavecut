"""Describe: da una frase a una playlist, con la libreria che resta a casa.

«Synth pop anni 80, solo versioni extended» è una domanda in tre pezzi:
un intervallo di anni, due etichette di genere, una parola da cercare nel
titolo. Qui c'è il modulo che li tiene — la `Query` — e la ricerca che lo
applica alla mappa. Chi COMPILA il modulo sta altrove, ed è in due:
`describe_lexicon` lo fa a regole, senza rete; `describe_llm` lo chiede a
un modello, mandandogli la frase e il vocabolario e niente altro. Tutti
e due tornano una `Query`, e da lì in poi la strada è una sola.

La ricerca è in due tempi. I **filtri duri** — anni, tempo, durata, parole
nel titolo — dicono chi può entrare. Le **etichette** — generi e mood, dal
vocabolario dei modelli Essentia — dicono chi entra per primo: i brani
che le portano più forte sono i semi, e il Radio Mix (`radio.tune`)
riempie il resto sugli embedding, restando dentro i filtri. Così un
«synth pop» prende anche il brano che SUONA synth pop ma che il modello
ha etichettato new wave — e un «anni 80» resta un anni 80.

Le etichette di una `Query` sono quelle della libreria, con la loro
grafia: `Vocabulary.of(frame)` le elenca, `Query.cleaned` riporta a
quella grafia ciò che arriva da fuori e scarta ciò che non esiste. Un
modello che inventa un genere non fa entrare niente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.analysis import radio
from core.analysis.duplicates import folded
from core.analysis.year_guess import MIN_CONFIDENCE
from core.analysis.years import FIRST_YEAR, LAST_YEAR

# Cento brani: uno scaffale per una serata, non una scaletta da suonare
# intera.
DEFAULT_SIZE = 100

# Quanto il Radio Mix si allontana da ciò che ha già preso: il valore di
# default della manopola «Variety» in Build a set.
DEFAULT_VARIETY = 0.5

# Le corse entro cui un tempo ha senso.
BPM_FLOOR, BPM_CEILING = 40.0, 300.0


def _pair(values, floor: float, ceiling: float, whole: bool = False):
    """Un intervallo (basso, alto) pulito, o None se non se ne cava uno."""
    if values is None:
        return None
    try:
        low, high = float(values[0]), float(values[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    low, high = max(floor, min(low, high)), min(ceiling, max(low, high))
    if low > high:
        return None
    return (int(low), int(high)) if whole else (low, high)


def _words(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass
class Vocabulary:
    """Le etichette che QUESTA libreria porta: generi e mood, i più
    frequenti per primi. È tutto ciò che di sé la libreria dice a chi
    legge la frase."""

    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, frame: pd.DataFrame) -> "Vocabulary":
        return cls(genres=_labels(frame, "genre_list", "genres"),
                   moods=_labels(frame, "mood_list", "moods"))

    def canonical(self, kind: str, name: str) -> str | None:
        """La grafia della libreria per un'etichetta scritta comunque, o
        None se la libreria non la porta."""
        wanted = folded(name)
        for label in getattr(self, kind):
            if folded(label) == wanted:
                return label
        return None


def _labels(frame: pd.DataFrame, lists: str, joined: str) -> list[str]:
    if lists in frame:
        tags = [t for row in frame[lists] for t in row if t]
    elif joined in frame:
        tags = [t for row in frame[joined].fillna("")
                for t in str(row).split("; ") if t]
    else:
        return []
    return list(pd.Series(tags).value_counts().index) if tags else []


@dataclass
class Query:
    """Il modulo compilato: cosa la frase chiede, in termini che la mappa
    sa applicare. Vuoto vuol dire "tutto"."""

    years: tuple[int, int] | None = None
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    bpm: tuple[float, float] | None = None
    title_words: list[str] = field(default_factory=list)
    min_minutes: float | None = None
    # Una riga di chi ha letto la frase: cosa ha capito, in parole.
    how_read: str = ""

    def is_empty(self) -> bool:
        return (self.years is None and not self.genres and not self.moods
                and self.bpm is None and not self.title_words
                and self.min_minutes is None)

    def to_dict(self) -> dict:
        return {"years": list(self.years) if self.years else None,
                "genres": list(self.genres), "moods": list(self.moods),
                "bpm": list(self.bpm) if self.bpm else None,
                "title_words": list(self.title_words),
                "min_minutes": self.min_minutes,
                "how_read": self.how_read}

    @classmethod
    def from_dict(cls, data) -> "Query":
        """Da un dizionario scritto da chiunque — un file, un modello —
        senza fidarsi: ciò che non si legge cade, non rompe."""
        data = data if isinstance(data, dict) else {}
        minutes = data.get("min_minutes")
        try:
            minutes = float(minutes) if minutes is not None else None
            if minutes is not None and not (minutes > 0):
                minutes = None
        except (TypeError, ValueError):
            minutes = None
        return cls(years=_pair(data.get("years"), FIRST_YEAR, LAST_YEAR,
                               whole=True),
                   genres=_words(data.get("genres")),
                   moods=_words(data.get("moods")),
                   bpm=_pair(data.get("bpm"), BPM_FLOOR, BPM_CEILING),
                   title_words=_words(data.get("title_words")),
                   min_minutes=minutes,
                   how_read=str(data.get("how_read") or ""))

    def cleaned(self, vocabulary: Vocabulary) -> "Query":
        """La stessa domanda con le etichette nella grafia della libreria,
        e senza quelle che la libreria non ha."""
        genres = [g for g in (vocabulary.canonical("genres", name)
                              for name in self.genres) if g]
        moods = [m for m in (vocabulary.canonical("moods", name)
                             for name in self.moods) if m]
        return Query(years=self.years, genres=list(dict.fromkeys(genres)),
                     moods=list(dict.fromkeys(moods)), bpm=self.bpm,
                     title_words=list(self.title_words),
                     min_minutes=self.min_minutes, how_read=self.how_read)


def summary(query: Query) -> str:
    """La lettura in una riga, da mostrare PRIMA della lista: «1980–1989 ·
    Synth-pop, New Wave · title has extended · ≥ 5.5 min»."""
    parts = []
    if query.years:
        low, high = query.years
        parts.append(f"{low}–{high}" if low != high else str(low))
    if query.genres:
        parts.append(", ".join(g.split(" - ")[-1] for g in query.genres))
    if query.moods:
        parts.append(", ".join(query.moods))
    if query.bpm:
        parts.append(f"{query.bpm[0]:.0f}–{query.bpm[1]:.0f} BPM")
    if query.title_words:
        parts.append("title has " + " / ".join(query.title_words))
    if query.min_minutes:
        parts.append(f"≥ {query.min_minutes:g} min")
    return " · ".join(parts) if parts else "everything"


# --------------------------------------------------------------------------
# la ricerca
# --------------------------------------------------------------------------

def _numbers(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def years_of(frame: pd.DataFrame) -> pd.Series:
    """L'anno su cui un filtro legge ogni brano: il tag se c'è, altrimenti
    la stima di Claude quando è abbastanza sicura (`year_guess` con
    `year_guess_conf` da `MIN_CONFIDENCE` in su), altrimenti niente."""
    year = _numbers(frame, "year")
    guess = _numbers(frame, "year_guess")
    confidence = _numbers(frame, "year_guess_conf")
    trusted = guess.where(confidence >= MIN_CONFIDENCE)
    return year.where(year.notna(), trusted)


def guessed_years(frame: pd.DataFrame) -> pd.Series:
    """Dove l'anno letto è una stima e non un tag."""
    return _numbers(frame, "year").isna() & years_of(frame).notna()


def _haystack(frame: pd.DataFrame) -> pd.Series:
    """Dove si cercano le parole del titolo: nome del file e titolo dei
    tag, con gli accenti piegati come in `matching_tracks`."""
    hay = frame["name"].fillna("").astype(str) if "name" in frame \
        else pd.Series("", index=frame.index)
    if "title" in frame:
        hay = hay + " " + frame["title"].fillna("").astype(str)
    return hay.map(folded)


def pool(frame: pd.DataFrame, query: Query) -> list[int]:
    """Chi passa i filtri duri: anni, tempo, durata, parole nel titolo.

    Un brano senza anno NON passa un filtro sugli anni: «anni 80» con
    dentro tutto ciò che non si sa datare non sarebbe più un anni 80. Vale
    il tag, o una stima di Claude abbastanza sicura (`years_of`). Quanti
    restino fuori lo dice `Match.no_year`, e quante stime siano entrate
    `Match.guessed`; il rimedio è `years_cli.py`. Un
    brano senza tempo invece passa un filtro sul tempo, come nei filtri
    della pagina: non sappiamo dove cade, e "no" sarebbe una risposta a
    una domanda che non è stata posta.
    """
    keep = pd.Series(True, index=frame.index)
    if query.years is not None:
        keep &= years_of(frame).between(*query.years)
    if query.bpm is not None:
        bpm = _numbers(frame, "bpm")
        keep &= bpm.isna() | bpm.between(*query.bpm)
    if query.min_minutes is not None:
        duration = _numbers(frame, "duration")
        keep &= duration >= query.min_minutes * 60
    if query.title_words:
        hay = _haystack(frame)
        hit = pd.Series(False, index=frame.index)
        for word in query.title_words:
            hit |= hay.str.contains(folded(word), regex=False)
        keep &= hit
    return [int(i) for i in keep[keep].index]


def _strength(tags: list[str], wanted: set[str]) -> float:
    """Quanto forte un brano porta una delle etichette chieste: le
    etichette sono in ordine di forza, la prima vale 1, la seconda ½…"""
    return max((1.0 / (n + 1) for n, tag in enumerate(tags) if tag in wanted),
               default=0.0)


def seeds(frame: pd.DataFrame, query: Query, candidates: list[int],
          song_of=None) -> list[int]:
    """I brani del pool che portano le etichette chieste, dal più forte al
    più debole; a parità, l'ordine della libreria. Con `song_of` le copie
    della stessa canzone contano una volta, la più forte."""
    genres, moods = set(query.genres), set(query.moods)
    if not genres and not moods:
        return []
    scored = []
    for i in candidates:
        score = 0.0
        if genres and "genre_list" in frame:
            score += _strength(frame.at[i, "genre_list"], genres)
        if moods and "mood_list" in frame:
            score += _strength(frame.at[i, "mood_list"], moods)
        if score > 0:
            scored.append((-score, i))
    scored.sort()
    out, songs = [], set()
    for _, i in scored:
        if song_of is not None:
            song = song_of(i)
            if song in songs:
                continue
            songs.add(song)
        out.append(i)
    return out


def spread(items: list[int], size: int) -> list[int]:
    """`size` elementi presi a passo regolare lungo `items`: un campione
    che copre la fila intera invece di fermarsi all'inizio."""
    if size <= 0:
        return []
    if len(items) <= size:
        return list(items)
    if size == 1:
        return [items[0]]
    return [items[round(k * (len(items) - 1) / (size - 1))]
            for k in range(size)]


@dataclass
class Match:
    """Cosa la ricerca ha trovato, e perché."""

    query: Query
    pool: list[int]        # chi passa i filtri duri
    seeds: list[int]       # chi porta le etichette, i primi a entrare
    tracks: list[int]      # la playlist, semi per primi
    no_year: int = 0       # brani tenuti fuori perché senza anno
    guessed: int = 0       # brani della playlist datati da una stima


def search(frame: pd.DataFrame, embeddings, query: Query,
           size: int = DEFAULT_SIZE, variety: float = DEFAULT_VARIETY,
           song_of=None) -> Match:
    """La playlist di una `Query`: fino a `size` brani.

    Prima i semi — chi porta le etichette, dal più forte — poi, se non
    bastano, il Radio Mix riempie sugli embedding dentro al pool. Senza
    etichette non c'è un gusto da inseguire: la playlist è un campione a
    passo regolare del pool ordinato per anno, così una decade si copre
    intera invece di fermarsi al suo primo anno.
    """
    candidates = pool(frame, query)
    no_year = 0
    if query.years is not None:
        no_year = int(years_of(frame).isna().sum())

    taken = seeds(frame, query, candidates, song_of)[:size]
    if not query.genres and not query.moods:
        year = years_of(frame).fillna(LAST_YEAR + 1)
        by_year = sorted(candidates, key=lambda i: (float(year.at[i]), i))
        return _matched(frame, query, candidates, [], spread(by_year, size),
                        no_year)

    tracks = list(taken)
    room = size - len(tracks)
    if room > 0 and taken and embeddings is not None and len(embeddings):
        seeded = set(taken)
        rest = [i for i in candidates if i not in seeded]
        tracks += radio.tune(embeddings, taken, pool=rest, k=room,
                             variety=variety, song_of=song_of)
    return _matched(frame, query, candidates, taken, tracks, no_year)


def _matched(frame, query, candidates, taken, tracks, no_year) -> Match:
    guessed = 0
    if query.years is not None and tracks:
        guessed = int(guessed_years(frame).loc[tracks].sum())
    return Match(query, candidates, taken, tracks, no_year, guessed)
