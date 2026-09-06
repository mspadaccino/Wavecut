"""Il lettore a modello: la frase a Claude, il modulo indietro.

Cosa parte: un testo fisso che spiega il compito e allega il vocabolario
di QUESTA libreria — le etichette di genere e di mood che i suoi brani
portano — più la frase dell'utente, com'è. Cosa NON parte: la libreria.
Nessun nome di file, nessun titolo, nessun embedding: il modello non sa
cosa l'utente possiede e non gli serve saperlo.

Cosa torna: la `Query`, e non una chiacchierata. Lo schema lo fissiamo
noi e l'API garantisce che la risposta lo rispetti (structured outputs);
poi `Query.cleaned` riporta le etichette alla grafia della libreria e
scarta quelle che il modello si è inventato. Un'etichetta inventata non fa
entrare niente.

Il testo fisso è sempre uguale per la stessa libreria, e sta in cache
lato API: dalla seconda lettura in poi la frase è quasi tutto ciò che si
paga — un centesimo circa. Le letture fatte si tengono su disco
(`Readings`): la stessa frase non si rimanda mai.

La chiave è dell'utente (`api_keys`). Senza chiave, senza rete, con la
chiave sbagliata o il credito finito, chi chiama riceve un
`ReadingFailed` con una riga da mostrare e passa al lettore a regole:
nessuna finestra di errore, nessuna funzione che sparisce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.analysis.describe import Query, Vocabulary, summary
from core.analysis.duplicates import folded
from core.analysis.user_files import user_dir

# Il modello e quanto pensarci: leggere una frase in un modulo è un compito
# piccolo, e `low` lo fa in un secondo o due.
DEFAULT_MODEL = "claude-opus-5"
EFFORT = "low"
# Il modulo sono duecento token; il resto è spazio per il pensiero, che
# conta anch'esso nel massimo.
MAX_TOKENS = 4096

# Quanto si aspetta la rete prima di passare al lessico: in sala non si
# aspetta di più.
TIMEOUT_SECONDS = 15.0

SYSTEM_PROMPT = """You turn a DJ's description of a playlist into a search form.

The DJ types a short phrase, in Italian or English, such as "synth pop anni 80,
solo versioni extended" or "90s eurodance floor fillers". You fill in the form
below. Everything you write must be derivable from the phrase; leave a field
empty (null or []) when the phrase says nothing about it.

Fields:
- years: [first, last], inclusive. A decade is its ten years: "80s" is
  [1980, 1989]. "dal 1985" is [1985, 2100]. null when no era is named.
- genres: labels chosen ONLY from the genre vocabulary below, spelled exactly
  as listed. Pick every label that fits the phrase (a macro genre word such as
  "rock" means every label of that family). [] if none fits.
- moods: labels chosen ONLY from the mood vocabulary below, exactly as listed.
- bpm: [low, high] tempo range, only when the phrase says something about
  tempo ("slow", "ballad", "under 100 bpm"); ballads are [40, 105]. Else null.
- title_words: words that must appear in the track title for the kind of
  version asked — "remix", "extended", "12\\"", "dub", "instrumental",
  "rework", "re-edit", "megamix". [] if the phrase does not ask for a version.
- min_minutes: minimum length in minutes when the phrase asks for long
  versions (5.5 for "extended"/"long versions" only if it does not already ask
  for the word in the title); else null.
- how_read: one short sentence, in the language of the phrase, saying how you
  read it.

Some DJ words mean an era and a sound together: "flash house" is late-80s to
early-90s house, freestyle and hi-NRG, years [1986, 1993]; "eurodance" is the
90s eurodance/euro house sound; "revibes" means modern re-edits and reworks
(title words rework, re-edit, edit, refix, bootleg).

Never invent a label. Never add a field the phrase does not support.
"""


class ReadingFailed(Exception):
    """La lettura a modello non è arrivata: il messaggio è la riga da
    mostrare, e chi chiama passa al lessico."""


def vocabulary_prompt(vocabulary: Vocabulary) -> str:
    """Il testo fisso: istruzioni più le etichette di questa libreria. Le
    etichette in ordine stabile, così il testo è identico da una lettura
    all'altra e la cache dell'API lo riconosce."""
    genres = "\n".join(f"- {g}" for g in sorted(vocabulary.genres))
    moods = "\n".join(f"- {m}" for m in sorted(vocabulary.moods))
    return (SYSTEM_PROMPT
            + "\nGenre vocabulary:\n" + (genres or "- (none)")
            + "\n\nMood vocabulary:\n" + (moods or "- (none)") + "\n")


def _schema():
    """Il modulo come modello pydantic, per gli structured outputs. Definito
    qui dentro perché pydantic arriva con `anthropic`, che è opzionale."""
    from pydantic import BaseModel

    class Form(BaseModel):
        years: list[int] | None = None
        genres: list[str] = []
        moods: list[str] = []
        bpm: list[float] | None = None
        title_words: list[str] = []
        min_minutes: float | None = None
        how_read: str = ""

    return Form


class ClaudeReader:
    """Legge la frase con Claude. `client` si può passare — è come si prova
    senza rete — altrimenti si costruisce dalla chiave."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 client=None, timeout: float = TIMEOUT_SECONDS) -> None:
        self.model = model
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise ReadingFailed("No API key: add yours under 🔑, or "
                                    "the phrase is read by the rules only.")
            try:
                import anthropic
            except ImportError as trouble:
                raise ReadingFailed("The `anthropic` package is not "
                                    "installed.") from trouble
            self._client = anthropic.Anthropic(api_key=self._api_key,
                                               timeout=self._timeout,
                                               max_retries=1)
        return self._client

    def read(self, text: str, vocabulary: Vocabulary) -> Query:
        text = text.strip()
        if not text:
            return Query(how_read="everything")
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                system=[{"type": "text", "text": vocabulary_prompt(vocabulary),
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": text}],
                output_format=_schema(),
            )
        except ReadingFailed:
            raise
        except Exception as trouble:                    # noqa: BLE001
            raise ReadingFailed(_explain(trouble)) from trouble
        if getattr(response, "stop_reason", None) == "refusal":
            raise ReadingFailed("The model declined to read this phrase.")
        form = getattr(response, "parsed_output", None)
        if form is None:
            raise ReadingFailed("The model answered without the form.")
        query = Query.from_dict(form.model_dump()).cleaned(vocabulary)
        if not query.how_read:
            query.how_read = summary(query)
        return query


def _explain(trouble: Exception) -> str:
    """Una riga per l'utente, dal tipo dell'errore: che cosa fare, non
    lo stack."""
    name = type(trouble).__name__
    if name == "AuthenticationError":
        return "The API key was refused: check it under 🔑."
    if name in ("PermissionDeniedError",):
        return "The API key has no permission for this model."
    if name in ("RateLimitError",):
        return "Too many requests for this key right now — try again in a moment."
    if name == "APITimeoutError":
        return "No answer from the model in time."
    if name == "APIConnectionError":
        return "No answer from the network."
    if name == "BadRequestError" and "credit" in str(trouble).lower():
        return "The key has no credit left."
    return f"The model could not be reached ({name})."


class Readings:
    """Le letture già fatte, su disco: `{frase: modulo}` in un JSON accanto
    ai preset. La chiave è la frase piegata — maiuscole, accenti e spazi
    in più non fanno una frase nuova."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else user_dir() / "readings.json"

    @staticmethod
    def key(text: str) -> str:
        return " ".join(folded(text).split())

    def _all(self) -> dict:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, text: str) -> Query | None:
        saved = self._all().get(self.key(text))
        return Query.from_dict(saved) if isinstance(saved, dict) else None

    def put(self, text: str, query: Query) -> None:
        everything = self._all()
        everything[self.key(text)] = query.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(everything, indent=2,
                                        ensure_ascii=False), "utf-8")

    def forget(self, text: str) -> None:
        everything = self._all()
        if self.key(text) in everything:
            del everything[self.key(text)]
            self.path.write_text(json.dumps(everything, indent=2,
                                            ensure_ascii=False), "utf-8")


# --------------------------------------------------------------------------
# la cura: Claude sceglie dentro una rosa, non dentro la libreria
# --------------------------------------------------------------------------

# Quanti candidati per ogni brano voluto: la ricerca locale ne porta tre
# volte tanti, e Claude tiene i migliori.
CANDIDATES_PER_PICK = 3

# Per quanti brani si chiede anche il perché: dieci righe si leggono, cento
# no, e ogni riga costa.
REASONS_FOR = 10

# Scegliere fra trecento righe non è leggere una frase: la risposta sono
# cento voci, e i quindici secondi del lettore non bastano. Due minuti, e
# spazio per le cento voci più il pensiero che le precede.
CURATE_TIMEOUT = 120.0
CURATE_MAX_TOKENS = 8192

CURATE_PROMPT = """You curate a DJ's playlist from a shortlist.

The DJ described the playlist in a phrase; the app read it into a form and
searched the library, and here is the shortlist it found — one track per
line, numbered, with what is known about each: title, artist, year, tempo,
key, the genre and mood labels a model gave it. Your job is to pick the best
{size} tracks for what the DJ asked, using what you know about these records:
which are the classics that fill a floor, which are the versions DJs play,
which do not belong despite passing the filters. Prefer tracks that fit the
phrase in spirit, not only in label. Keep the DJ's language for the reasons.

Answer with the picks in the order you would recommend them, best first, as
JSON: {{"picks": [{{"id": <number>, "why": <short reason or null>}}]}}.
Give a reason for the first {reasons} only; null for the rest. Pick at most
{size}. Never invent an id; if fewer than {size} deserve the list, pick fewer.
"""


@dataclass
class Curation:
    """Cosa Claude ha tenuto: le posizioni scelte nell'ordine suo, e il
    perché per le prime."""

    picks: list[int]
    reasons: dict[int, str] = field(default_factory=dict)


def candidate_line(number: int, row) -> str:
    """Un candidato in una riga: quello che si sa, corto."""
    title = str(row.get("title") or "").strip()
    artist = str(row.get("artist") or "").strip()
    name = f"{artist} - {title}".strip(" -") or str(row.get("name") or "")
    pieces = [f"{number}. {name}"]
    for key, label in (("year", ""), ("bpm", " bpm"), ("camelot", "")):
        value = row.get(key)
        if value is None or value != value or not str(value):
            continue                                    # manca, o è nan
        pieces.append(f"{float(value):.0f}{label}" if key in ("year", "bpm")
                      else f"{value}{label}")
    for key in ("genres", "moods"):
        value = str(row.get(key) or "").strip()
        if value:
            pieces.append(value)
    return " | ".join(pieces)


def _curation_schema():
    from pydantic import BaseModel

    class Pick(BaseModel):
        id: int
        why: str | None = None

    class Picks(BaseModel):
        picks: list[Pick] = []

    return Picks


class ClaudeCurator(ClaudeReader):
    """Sceglie dentro una rosa. Stessa chiave del lettore, più tempo."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 client=None, timeout: float = CURATE_TIMEOUT) -> None:
        super().__init__(api_key=api_key, model=model, client=client,
                         timeout=timeout)

    def curate(self, phrase: str, query: Query, frame,
               candidates: list[int], size: int) -> Curation:
        """I `size` migliori fra `candidates` (posizioni nel frame) per la
        frase, nell'ordine di Claude. Un id inventato cade; nessuna
        risposta o un guasto sono un `ReadingFailed`."""
        if not candidates or size <= 0:
            return Curation(picks=[])
        listed = "\n".join(candidate_line(n + 1, frame.loc[i])
                           for n, i in enumerate(candidates))
        asked = (f"The DJ asked: {phrase.strip()}\n"
                 f"Read as: {query.how_read or summary(query)}\n\n"
                 f"Shortlist:\n{listed}")
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=CURATE_MAX_TOKENS,
                output_config={"effort": EFFORT},
                system=CURATE_PROMPT.format(size=size, reasons=REASONS_FOR),
                messages=[{"role": "user", "content": asked}],
                output_format=_curation_schema(),
            )
        except ReadingFailed:
            raise
        except Exception as trouble:                    # noqa: BLE001
            raise ReadingFailed(_explain(trouble)) from trouble
        if getattr(response, "stop_reason", None) == "refusal":
            raise ReadingFailed("The model declined to curate this list.")
        form = getattr(response, "parsed_output", None)
        if form is None:
            raise ReadingFailed("The model answered without the picks.")
        picks, reasons, seen = [], {}, set()
        for pick in form.picks:
            number = int(pick.id)
            if not 1 <= number <= len(candidates) or number in seen:
                continue
            seen.add(number)
            index = candidates[number - 1]
            picks.append(index)
            if pick.why:
                reasons[index] = str(pick.why).strip()
            if len(picks) >= size:
                break
        return Curation(picks=picks, reasons=reasons)
