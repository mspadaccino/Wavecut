"""Il lettore a regole: da una frase a una `Query`, senza rete.

È il lettore che c'è sempre — senza chiave, senza connessione, in sala
con il wifi morto — e la scala su cui si misura quello a modello: se il
modello legge peggio di queste regole, non serve. Capisce le parole che
gli sono state insegnate, e sono tre famiglie:

- le **epoche**: «70s», «anni '80», «eighties», «1985-1992», «dal 1983»;
- le **etichette** dei modelli, per nome: «synth pop», «new wave», «italo
  disco», «rock» (il macro genere, tutte le sue foglie), «happy», «dark»;
- le **parole del mestiere**, che non stanno in nessun modello e vanno
  scritte a mano in `ALIASES`: «ballad», «extended», «12 inch», «flash
  house», «eurodance», «revibes». È il posto dove il DJ insegna al
  lettore il suo vocabolario, ed è fatto per crescere.

Italiano e inglese insieme: chi scrive «anni 80 remix lenti» e chi scrive
«80s slow remixes» chiede la stessa cosa. Le etichette si confrontano
COMPATTE — senza spazi, trattini, accenti né maiuscole — così «synthpop»,
«synth-pop» e «Synth Pop» sono la stessa parola; e sempre per intero,
così «house» prende House e non Deep House, che è un'altra domanda.
"""

from __future__ import annotations

import re

from core.analysis.describe import (BPM_FLOOR, Query, Vocabulary, summary)
from core.analysis.duplicates import folded

# Le epoche dette a parole. Il valore è il primo anno della decade.
DECADE_WORDS = {
    "fifties": 1950, "cinquanta": 1950,
    "sixties": 1960, "sessanta": 1960,
    "seventies": 1970, "settanta": 1970,
    "eighties": 1980, "ottanta": 1980,
    "nineties": 1990, "novanta": 1990,
    "noughties": 2000, "duemila": 2000,
}

# I macro generi detti in italiano, o con una parola sola.
PARENT_WORDS = {
    "elettronica": "Electronic", "electronica": "Electronic",
    "classica": "Classical", "latina": "Latin", "latino": "Latin",
    "rap": "Hip Hop", "hiphop": "Hip Hop",
    "funky": "Funk / Soul",
}

# I mood detti in italiano, verso l'etichetta inglese del modello.
MOOD_WORDS = {
    "allegro": "happy", "allegra": "happy", "felice": "happy",
    "triste": "sad", "malinconico": "melancholic",
    "malinconica": "melancholic", "romantico": "romantic",
    "romantica": "romantic", "cupo": "dark", "cupa": "dark",
    "oscuro": "dark", "scuro": "dark", "festa": "party",
    "estate": "summer", "estivo": "summer", "estiva": "summer",
    "energico": "energetic", "energica": "energetic",
    "rilassato": "relaxing", "rilassante": "relaxing",
    "epico": "epic", "epica": "epic", "sognante": "dreamy",
    "aggressivo": "aggressive", "dolce": "sweet", "calmo": "calm",
    "calma": "calm", "sensuale": "sexy",
}

# Le parole del mestiere. Ogni voce: le forme con cui si dice (compatte,
# come si confrontano) e i pezzi di Query che accende. `genres` sono
# FOGLIE di etichetta, e quelle che questa libreria non ha cadono da sole.
ALIASES: list[tuple[tuple[str, ...], dict]] = [
    (("ballad", "ballads", "ballata", "ballate", "lenti", "lento", "slow",
      "slows"),
     {"bpm": (BPM_FLOOR, 105.0), "moods": ["romantic", "melancholic"]}),
    (("remix", "remixes", "remixed", "remixata", "remixati"),
     {"title_words": ["remix"]}),
    (("extended", "extendeds"), {"title_words": ["extended"]}),
    (("12inch", "12\"", "12''", "maxi", "maxisingle", "12pollici"),
     {"title_words": ["12\"", "12''", "12 inch", "maxi"]}),
    (("rework", "reworks", "reedit", "reedits", "edit", "edits"),
     {"title_words": ["rework", "re-edit", "reedit", "edit"]}),
    (("dub", "dubs"), {"title_words": ["dub"]}),
    (("instrumental", "instrumentals", "strumentale", "strumentali"),
     {"title_words": ["instrumental"]}),
    (("mastermix", "mastermixes", "megamix", "megamixes"),
     {"title_words": ["mastermix", "megamix"]}),
    (("longversion", "longversions", "versionelunga", "versionilunghe",
      "lunghe", "lunghi"),
     {"min_minutes": 5.5}),
    (("eurodance",), {"genres": ["Eurodance", "Euro House", "Italodance"]}),
    (("italo",), {"genres": ["Italo-Disco", "Italo House", "Italodance"]}),
    (("dance",), {"genres": ["Eurodance", "Euro House", "Italodance",
                             "Dance-pop", "Hi NRG"]}),
    (("flashhouse",),
     {"genres": ["House", "Freestyle", "Hi NRG", "Italo House", "Euro House",
                 "Garage House"],
      "years": (1986, 1993)}),
    (("revibes", "revibe", "revibed"),
     {"title_words": ["rework", "re-edit", "reedit", "revibe", "refix",
                      "bootleg"]}),
]

_YEAR = r"(?:19|20)\d\d"
_RANGE = re.compile(rf"\b({_YEAR})\s*(?:-|–|—|to|al|a|e|and|/)\s*({_YEAR})\b")
_SINGLE = re.compile(rf"\b({_YEAR})\b")
_FROM = re.compile(rf"\b(?:dal|from|since|after|dopo il|dopo)\s+({_YEAR})\b")
_UNTIL = re.compile(rf"\b(?:fino al|until|before|prima del|up to)\s+({_YEAR})\b")
_LONG_DECADE = re.compile(r"\b(19|20)(\d)0s\b")
# «anni 2000», «anni 1980»: l'anno per intero dopo «anni» è una decade.
_ANNI_FULL = re.compile(r"\banni\s+(19|20)(\d)0\b")
# «anni 80», «anni '80», «'80», «80s», «80's»: mai un «80» nudo, che è
# un numero (80 bpm) prima che una decade.
_SHORT_DECADES = (re.compile(r"\banni\s*'?(\d)0\b"),
                  re.compile(r"(?:^|[^\w])'(\d)0(?:s|'s)?\b"),
                  re.compile(r"\b(\d)0(?:s|'s)\b"))
_BPM_RANGE = re.compile(r"\b(\d{2,3})\s*(?:-|–|—|to|a|e|and|/)\s*(\d{2,3})\s*bpm\b")
_BPM_UNDER = re.compile(r"\b(?:sotto i|sotto|under|below|less than|meno di|max)\s*(\d{2,3})\s*bpm\b")
_BPM_OVER = re.compile(r"\b(?:sopra i|sopra|over|above|more than|piu di|oltre i|oltre|min)\s*(\d{2,3})\s*bpm\b")
_MINUTES = re.compile(r"\b(?:oltre i|oltre|over|almeno|at least|piu di|more than|longer than|min)\s*(\d+(?:[.,]\d+)?)\s*(?:min|minuti|minutes|minute)\b")


def compact(text: str) -> str:
    """Il testo come si confronta: minuscolo, senza accenti, senza niente
    che non sia lettera, cifra o apice (che «12"» lo vuole)."""
    return re.sub(r"[^a-z0-9\"']", "", folded(text))


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9\"']+", folded(text)) if t]


def _grams(tokens: list[str], up_to: int = 3) -> dict[str, list[tuple[int, int]]]:
    """Ogni sequenza di una, due o tre parole, compatta, con dove sta
    (inizio, lunghezza): è contro queste che un'etichetta a più parole si
    riconosce, e la posizione dice quali parole ha preso."""
    out: dict[str, list[tuple[int, int]]] = {}
    for n in range(1, up_to + 1):
        for k in range(len(tokens) - n + 1):
            out.setdefault("".join(tokens[k:k + n]), []).append((k, n))
    return out


def _decade(first: int) -> tuple[int, int]:
    return first, first + 9


def _short_decade(digit: str) -> tuple[int, int]:
    # «70s» è il Novecento, «00s» e «10s» e «20s» sono il Duemila.
    first = int(digit) * 10
    return _decade(1900 + first if first >= 30 else 2000 + first)


def years_in(text: str) -> tuple[int, int] | None:
    """Gli anni che la frase nomina, fusi in un intervallo solo: «70s e
    80s» è 1970–1989, «dal 1985» è dal 1985 a oggi."""
    text = folded(text)
    spans: list[tuple[int, int]] = []
    for low, high in _RANGE.findall(text):
        spans.append((int(low), int(high)))
    ranged = _RANGE.sub(" ", text)
    for year in _FROM.findall(ranged):
        spans.append((int(year), 2100))
    for year in _UNTIL.findall(ranged):
        spans.append((1900, int(year)))
    loose = _FROM.sub(" ", _UNTIL.sub(" ", ranged))
    for century, digit in _ANNI_FULL.findall(loose):
        spans.append(_decade(int(century) * 100 + int(digit) * 10))
    loose = _ANNI_FULL.sub(" ", loose)
    for year in _SINGLE.findall(_LONG_DECADE.sub(" ", loose)):
        spans.append((int(year), int(year)))
    for century, digit in _LONG_DECADE.findall(text):
        spans.append(_decade(int(century) * 100 + int(digit) * 10))
    short = _LONG_DECADE.sub(" ", _SINGLE.sub(" ", loose))
    for pattern in _SHORT_DECADES:
        for digit in pattern.findall(short):
            spans.append(_short_decade(digit))
    for word, first in DECADE_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            spans.append(_decade(first))
    if not spans:
        return None
    low = min(s[0] for s in spans)
    high = max(s[1] for s in spans)
    return (low, min(high, 2100)) if low <= high else (high, low)


def _bpm_in(text: str) -> tuple[float, float] | None:
    text = folded(text)
    found = _BPM_RANGE.search(text)
    if found:
        low, high = sorted((float(found.group(1)), float(found.group(2))))
        return low, high
    under, over = _BPM_UNDER.search(text), _BPM_OVER.search(text)
    if under or over:
        return (float(over.group(1)) if over else BPM_FLOOR,
                float(under.group(1)) if under else 300.0)
    return None


def _minutes_in(text: str) -> float | None:
    found = _MINUTES.search(folded(text))
    return float(found.group(1).replace(",", ".")) if found else None


def _leaf(label: str) -> str:
    return label.split(" - ")[-1]


def _parent_words(label: str) -> list[str]:
    """Le parole con cui si dice il macro genere: «Funk / Soul» si dice
    «funk», «soul» o «funksoul»."""
    parent = label.split(" - ")[0]
    parts = [compact(p) for p in parent.split("/")]
    return [p for p in parts if p] + [compact(parent)]


def _said(grams: dict, form: str) -> list[tuple[int, int]]:
    """Dove la frase dice `form`, al singolare o al plurale."""
    return grams.get(form, []) + grams.get(form + "s", [])


def genres_in(text: str, vocabulary: Vocabulary) -> list[str]:
    return _genres_in(text, vocabulary)[0]


def _genres_in(text: str, vocabulary: Vocabulary) -> tuple[list[str], set[str]]:
    """Le etichette di genere che la frase nomina, per foglia intera o
    per macro genere, nell'ordine del vocabolario (i più frequenti
    prima) — e le parole della frase che nessuna foglia ha preso.

    Prima le foglie, poi i macro generi sulle parole che le foglie non
    hanno preso: in «synth pop» il «pop» è metà di Synth-pop, non il
    macro genere Pop, e in «italo disco» il «disco» è metà di Italo-Disco.
    """
    tokens = _tokens(text)
    grams = _grams(tokens)
    places = {label: _said(grams, compact(_leaf(label)))
              + _said(grams, compact(label)) for label in vocabulary.genres}
    # Prese sono le parole di una foglia a PIÙ parole, e si prendono tutte
    # prima di guardare le foglie di una parola sola: in «italo house» la
    # «house» è metà di Italo House, e House non entra. «ballad», che è
    # una foglia intera, resta libera: la voce «ballad» del mestiere — il
    # tempo lento, i mood — si accende con lei.
    taken: set[int] = set()
    for found in places.values():
        for start, n in found:
            if n > 1:
                taken.update(range(start, start + n))
    out = [label for label, found in places.items()
           if any(n > 1 or start not in taken for start, n in found)]
    free = {tokens[k] for k in range(len(tokens)) if k not in taken}
    wanted_parents = {PARENT_WORDS[word] for word in free if word in PARENT_WORDS}
    for label in vocabulary.genres:
        if label in out:
            continue
        parent = label.split(" - ")[0]
        if parent in wanted_parents or any(
                word in free or word + "s" in free
                for word in _parent_words(label)):
            out.append(label)
    return out, free


def moods_in(text: str, vocabulary: Vocabulary) -> list[str]:
    grams = _grams(_tokens(text))
    said = {MOOD_WORDS[g] for g in grams if g in MOOD_WORDS}
    return [m for m in vocabulary.moods
            if compact(m) in grams or m in said]


def _aliases_in(text: str, free: set[str]) -> list[dict]:
    """Le voci del mestiere che la frase dice. Una forma di una parola sola
    vale solo se quella parola è libera: in «italo house» l'«italo» è metà
    di Italo House, non la voce «italo»."""
    tokens = _tokens(text)
    grams = _grams(tokens)
    found = []
    for forms, pieces in ALIASES:
        for form in forms:
            if form in grams and (form in free or form not in set(tokens)):
                found.append(pieces)
                break
    return found


def _leaves_to_labels(leaves: list[str], vocabulary: Vocabulary) -> list[str]:
    wanted = {compact(leaf) for leaf in leaves}
    return [label for label in vocabulary.genres
            if compact(_leaf(label)) in wanted]


def _merge_spans(*spans):
    spans = [s for s in spans if s]
    if not spans:
        return None
    return (min(s[0] for s in spans), max(s[1] for s in spans))


def read(text: str, vocabulary: Vocabulary) -> Query:
    """La frase letta a regole. Sempre una Query, magari vuota: una frase
    che il lessico non capisce vuol dire «tutto», e la riga di lettura
    lo dice."""
    genres, free = _genres_in(text, vocabulary)
    moods = moods_in(text, vocabulary)
    years = years_in(text)
    bpm = _bpm_in(text)
    title_words: list[str] = []
    minutes = _minutes_in(text)
    for pieces in _aliases_in(text, free):
        genres += [g for g in _leaves_to_labels(pieces.get("genres", []),
                                                 vocabulary) if g not in genres]
        moods += [m for m in pieces.get("moods", [])
                  if m in vocabulary.moods and m not in moods]
        years = _merge_spans(years, pieces.get("years"))
        if bpm is None and pieces.get("bpm"):
            bpm = pieces["bpm"]
        title_words += [w for w in pieces.get("title_words", [])
                        if w not in title_words]
        if pieces.get("min_minutes") and minutes is None:
            minutes = pieces["min_minutes"]
    query = Query(years=years, genres=genres, moods=moods, bpm=bpm,
                  title_words=title_words, min_minutes=minutes)
    query.how_read = summary(query)
    return query
