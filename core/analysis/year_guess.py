"""L'anno stimato da Claude, per i brani che i tag non datano.

Metà di una libreria da DJ non porta un anno: compilation rippate senza
tag, file rinominati, promo. L'anno però è la cosa che un modello di
linguaggio SA: «New Order - Blue Monday» è il 1983 per chiunque abbia
letto abbastanza. Qui si chiede a Claude, per ogni brano senza anno,
l'anno di uscita ORIGINALE della registrazione — non della ristampa, non
della compilation — insieme a quanto ne è sicuro.

Si passa per la Batch API: metà prezzo, e un lavoro che dura un'ora non
va tenuto aperto in una finestra. Il flusso è in due tempi, come il job
della mappa: `--submit` manda le richieste e segna il lotto; `--collect`,
quando il lotto è finito, scrive le risposte sulle righe. Fra i due si
può chiudere tutto.

Cosa parte per ogni brano: titolo e artista dai tag, il nome del file e
della cartella. Niente altro. Quaranta brani per richiesta, così il
prologo si paga una volta ogni quaranta e non ogni uno.

**La stima è una stima.** Va in un campo suo, `year_guess`, con la sua
fiducia in `year_guess_conf`; il campo `year` dei tag non si tocca. Chi
filtra (Crate Buddy) legge il tag se c'è, la stima se è abbastanza sicura,
e dice quante stime ha usato. Un brano già chiesto porta il campo, anche
a `None`: non si richiede a ogni giro.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.analysis.years import FIRST_YEAR, LAST_YEAR

# Sonnet: datare un disco è conoscenza, non ragionamento, e a quarantamila
# brani il prezzo conta più della sfumatura. `--model` per cambiarlo.
DEFAULT_MODEL = "claude-sonnet-5"
PER_REQUEST = 40
MAX_TOKENS = 4096

# Sotto questa fiducia una stima non filtra: meglio "non so" di un anno
# tirato a caso che manda un brano nel decennio sbagliato.
MIN_CONFIDENCE = 0.6

SYSTEM_PROMPT = """You date recordings for a DJ's library.

You get a numbered list of tracks, one per line, with whatever is known: the
title and artist from the file's tags, and the file and folder names, which
often carry the artist and title when the tags do not. For each track give
the year the recording was ORIGINALLY released — the original single or album,
not a reissue, a remaster or the compilation it was taken from. A remix or an
extended version of a track is dated by the year that version came out, when
you know it; otherwise by the original.

Answer with one object per track, in the same order, as JSON:
{"tracks": [{"id": <number>, "year": <int or null>, "confidence": <0..1>}]}

confidence is how sure you are: 1.0 for a famous record you know for certain,
around 0.5 when the artist and title are clear but the year is a good guess,
0.2 or less when you are reading tea leaves. When you cannot tell the track
at all — an unnamed file, a white label, a bootleg — use null and 0.

Never invent an artist or title. Never skip an id.
"""

# Lo schema della risposta: l'API garantisce che arrivi così.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "year": {"type": ["integer", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "year", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tracks"],
    "additionalProperties": False,
}


def candidates(rows: list[dict]) -> list[int]:
    """Le posizioni delle righe senza anno dai tag e mai chieste."""
    return [i for i, row in enumerate(rows)
            if row.get("year") is None and "year_guess" not in row]


def line(number: int, row: dict) -> str:
    """Un brano in una riga, per la lista: quello che si sa, e basta."""
    title = str(row.get("title") or "").strip()
    artist = str(row.get("artist") or "").strip()
    path = Path(str(row.get("path") or ""))
    pieces = [f"{number}."]
    if title or artist:
        pieces.append(f"{artist} - {title}".strip(" -"))
    pieces.append(f"file: {path.stem}")
    if path.parent.name:
        pieces.append(f"folder: {path.parent.name}")
    return " | ".join(pieces)


def chunks(positions: list[int], size: int = PER_REQUEST) -> list[list[int]]:
    size = max(1, size)
    return [positions[k:k + size] for k in range(0, len(positions), size)]


def request(custom_id: str, rows: list[dict], positions: list[int],
            model: str = DEFAULT_MODEL) -> dict:
    """Una richiesta del lotto: `positions` sono le righe, numerate da 1 nel
    testo — l'id che torna indietro è quel numero, e `positions[id - 1]`
    è la riga."""
    listed = "\n".join(line(n + 1, rows[i]) for n, i in enumerate(positions))
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
            "system": [{"type": "text", "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": listed}],
        },
    }


def parse_answer(text: str) -> dict[int, tuple[int | None, float]]:
    """`{id: (anno, fiducia)}` da una risposta, tollerando quello che si
    può tollerare: un anno fuori scala diventa None, una fiducia fuori
    da 0..1 si stringe. Una risposta illeggibile è un dizionario vuoto."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        found = re.search(r"\{.*\}", str(text or ""), re.S)
        if not found:
            return {}
        try:
            data = json.loads(found.group(0))
        except ValueError:
            return {}
    out: dict[int, tuple[int | None, float]] = {}
    for item in (data.get("tracks") or []) if isinstance(data, dict) else []:
        try:
            number = int(item["id"])
            year = item.get("year")
            year = int(year) if year is not None else None
            confidence = float(item.get("confidence") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if year is not None and not FIRST_YEAR <= year <= LAST_YEAR:
            year = None
        out[number] = (year, max(0.0, min(1.0, confidence)))
    return out


def apply(rows: list[dict], numbered: dict[int, int],
          answers: dict[int, tuple[int | None, float]]) -> int:
    """Scrive le stime sulle righe: `numbered` dice quale riga porta ogni
    numero della lista, `answers` cosa il modello ha detto di ognuno.
    Torna quante righe hanno ricevuto un anno. Chi non ha risposta resta
    da chiedere."""
    dated = 0
    for number, i in numbered.items():
        if number not in answers:
            continue
        year, confidence = answers[number]
        rows[i]["year_guess"] = year
        rows[i]["year_guess_conf"] = round(confidence, 2)
        if year is not None:
            dated += 1
    return dated


# --------------------------------------------------------------------------
# il lotto, su disco: quali richieste, per quali righe
# --------------------------------------------------------------------------

@dataclass
class Lot:
    """Un lotto mandato: l'id del batch e, per ogni richiesta, i percorsi
    delle righe nell'ordine numerato. Per percorso e non per posizione,
    perché la mappa fra il submit e il collect può cambiare fila."""

    batch_id: str
    model: str
    requests: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "model": self.model,
                "requests": self.requests}

    @classmethod
    def from_dict(cls, data: dict) -> "Lot":
        return cls(batch_id=str(data["batch_id"]),
                   model=str(data.get("model") or DEFAULT_MODEL),
                   requests={str(k): [str(p) for p in v]
                             for k, v in (data.get("requests") or {}).items()})

    @property
    def tracks(self) -> int:
        return sum(len(paths) for paths in self.requests.values())


def lots_dir(store_dir: Path) -> Path:
    return store_dir.parent / "year_guess"


def save_lot(store_dir: Path, lot: Lot) -> Path:
    folder = lots_dir(store_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{lot.batch_id}.json"
    path.write_text(json.dumps(lot.to_dict(), indent=1), "utf-8")
    return path


def pending_lots(store_dir: Path) -> list[Lot]:
    folder = lots_dir(store_dir)
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            out.append(Lot.from_dict(json.loads(path.read_text("utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return out


def forget_lot(store_dir: Path, lot: Lot) -> None:
    try:
        (lots_dir(store_dir) / f"{lot.batch_id}.json").unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------
# con l'API
# --------------------------------------------------------------------------

def submit(client, store, model: str = DEFAULT_MODEL,
           per_request: int = PER_REQUEST, limit: int = 0) -> Lot | None:
    """Manda un lotto per le righe da chiedere. `limit` ferma a N brani
    (una prova prima del lotto vero). None se non c'è niente da chiedere."""
    todo = candidates(store.rows)
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        return None
    rows = store.rows
    requests, by_id = [], {}
    for n, positions in enumerate(chunks(todo, per_request)):
        custom_id = f"years-{n:05d}"
        requests.append(request(custom_id, rows, positions, model))
        by_id[custom_id] = [rows[i]["path"] for i in positions]
    batch = client.messages.batches.create(requests=requests)
    return Lot(batch_id=batch.id, model=model, requests=by_id)


@dataclass
class Collected:
    ended: bool
    answered: int = 0      # richieste con una risposta letta
    dated: int = 0         # righe che hanno ricevuto un anno
    failed: int = 0        # richieste senza risposta (errore, scadute)
    processing: int = 0    # richieste ancora in lavorazione, se non finito


def collect(client, store, lot: Lot) -> Collected:
    """Legge un lotto finito e scrive le stime sulle righe, per percorso.
    Se il lotto non è finito, dice a che punto è e non tocca niente."""
    batch = client.messages.batches.retrieve(lot.batch_id)
    if batch.processing_status != "ended":
        counts = getattr(batch, "request_counts", None)
        return Collected(ended=False,
                         processing=int(getattr(counts, "processing", 0) or 0))

    at_path = {row["path"]: i for i, row in enumerate(store.rows)}
    out = Collected(ended=True)
    for result in client.messages.batches.results(lot.batch_id):
        paths = lot.requests.get(result.custom_id)
        if paths is None:
            continue
        if result.result.type != "succeeded":
            out.failed += 1
            continue
        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), "")
        # Il numero nella lista è la posizione nel lotto; la riga si ritrova
        # per percorso, e un brano sparito dalla mappa perde la sua stima.
        numbered = {n + 1: at_path[p] for n, p in enumerate(paths)
                    if p in at_path}
        out.dated += apply(store.rows, numbered, parse_answer(text))
        out.answered += 1
    if out.answered:
        store.rewrite()
    return out


# --------------------------------------------------------------------------
# la via a mano: un file di testo per la chat, la risposta reimportata
# --------------------------------------------------------------------------
#
# L'API si paga a consumo; la chat di Claude sta in un abbonamento. Per una
# libreria propria, una volta, la strada è: l'app scrive i brani in file di
# testo con la consegna in testa, il DJ li dà alla chat e salva la risposta,
# l'app la rilegge. Manuale, ma gratis. L'app NON guida la chat da sé: non
# si fa, e i termini d'uso lo vietano.

# Quanti brani per file: la chat risponde con una riga per brano, e oltre
# qualche centinaio taglia o si stanca.
PER_FILE = 250

CHAT_INSTRUCTIONS = """Below is a numbered list of tracks from a DJ's library, one per line, with what
is known about each: the title and artist from the file's tags when they exist,
and the file and folder names, which often carry the artist and title.

For EVERY line, answer with one line in this exact form, nothing else:

<id> | <year> | <confidence>

- year: the year the recording was ORIGINALLY released — the original single
  or album, not a reissue, a remaster or the compilation it was taken from. A
  remix or extended version is dated by the year that version came out when
  you know it, otherwise by the original. Write - when you cannot tell.
- confidence: how sure you are, from 0 to 1. 1.0 for a famous record you know
  for certain; about 0.5 when the artist and title are clear but the year is a
  good guess; 0.2 or less when you are reading tea leaves; 0 with - for a track
  you cannot identify at all.

Keep the ids exactly as given, one answer per id, in order. No commentary.

"""

_ANSWER_LINE = re.compile(
    r"^\s*(\d+)\s*[|;,\t]\s*(\d{4}|-|null|none|\?)\s*(?:[|;,\t]\s*([01](?:[.,]\d+)?|[.,]\d+))?",
    re.I)


def export_text(rows: list[dict], positions: list[int]) -> str:
    """Il file per la chat: la consegna, poi i brani numerati da 1."""
    listed = "\n".join(line(n + 1, rows[i]) for n, i in enumerate(positions))
    return CHAT_INSTRUCTIONS + listed + "\n"


def export(store, folder: Path, per_file: int = PER_FILE,
           limit: int = 0) -> Lot | None:
    """Scrive i file di testo per i brani da chiedere e segna il lotto,
    come `submit`: la risposta si reimporta per nome di file."""
    todo = candidates(store.rows)
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    rows = store.rows
    lot = Lot(batch_id=f"chat-{folder.name}", model="chat")
    for n, positions in enumerate(chunks(todo, per_file)):
        name = f"years-{n + 1:03d}"
        (folder / f"{name}.txt").write_text(export_text(rows, positions), "utf-8")
        lot.requests[name] = [rows[i]["path"] for i in positions]
    return lot


def parse_chat_answer(text: str) -> dict[int, tuple[int | None, float]]:
    """`{id: (anno, fiducia)}` dalle righe «id | anno | fiducia» che la chat
    ha scritto, saltando tutto il resto. Una riga senza fiducia vale 0.5:
    un anno detto senza dire quanto se ne è sicuri non filtra."""
    out: dict[int, tuple[int | None, float]] = {}
    for raw in str(text or "").splitlines():
        found = _ANSWER_LINE.match(raw)
        if not found:
            continue
        number = int(found.group(1))
        year_text = found.group(2)
        year = int(year_text) if year_text.isdigit() else None
        if year is not None and not FIRST_YEAR <= year <= LAST_YEAR:
            year = None
        if found.group(3) is not None:
            confidence = float(found.group(3).replace(",", "."))
        else:
            confidence = 0.0 if year is None else 0.5
        out[number] = (year, max(0.0, min(1.0, confidence)))
    return out


def import_answer(store, lot: Lot, name: str, text: str) -> int:
    """Le risposte di un file sulle righe, per percorso. Torna quante
    righe hanno ricevuto un anno; -1 se `name` non è del lotto."""
    paths = lot.requests.get(name)
    if paths is None:
        return -1
    at_path = {row["path"]: i for i, row in enumerate(store.rows)}
    numbered = {n + 1: at_path[p] for n, p in enumerate(paths) if p in at_path}
    dated = apply(store.rows, numbered, parse_chat_answer(text))
    return dated


# --------------------------------------------------------------------------
# subito, a gruppi: la coda del job della mappa
# --------------------------------------------------------------------------
#
# Chi ha la chiave non fa backfill: alla fine del job, i brani appena
# entrati senza anno si chiedono a Claude a gruppi di quaranta con l'API
# normale, sincrona — a listino pieno, ma per i brani nuovi di una serata
# sono centesimi, e la risposta arriva prima che la mappa si ricarichi.

@dataclass
class Asked:
    asked: int = 0         # brani mandati
    dated: int = 0         # brani che hanno ricevuto un anno
    trouble: str = ""      # perché ci si è fermati, se ci si è fermati


def ask(client, store, model: str = DEFAULT_MODEL,
        per_request: int = PER_REQUEST, limit: int = 0,
        on_progress=None) -> Asked:
    """Chiede l'anno dei brani senza, a gruppi, e lo scrive sulle righe.

    Al primo guasto si ferma e lo dice: i brani non ancora chiesti non
    portano il campo e restano da chiedere alla prossima volta, o al
    batch, o alla chat. Le righe si riscrivono alla fine, e ogni dieci
    gruppi: un job interrotto a metà tiene quello che ha.
    """
    todo = candidates(store.rows)
    if limit > 0:
        todo = todo[:limit]
    out = Asked()
    if not todo:
        return out
    rows = store.rows
    groups = chunks(todo, per_request)
    for n, positions in enumerate(groups):
        params = request(f"now-{n}", rows, positions, model)["params"]
        try:
            response = client.messages.create(**params)
        except Exception as trouble:                    # noqa: BLE001
            out.trouble = f"{type(trouble).__name__}: {trouble}"[:200]
            break
        text = next((b.text for b in response.content if b.type == "text"), "")
        answers = parse_answer(text)
        numbered = {k + 1: i for k, i in enumerate(positions)}
        out.dated += apply(rows, numbered, answers)
        out.asked += len(positions)
        if on_progress:
            on_progress(out.asked, len(todo))
        if (n + 1) % 10 == 0:
            store.rewrite()
    if out.asked:
        store.rewrite()
    return out
