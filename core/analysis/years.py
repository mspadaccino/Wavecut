"""L'anno di un brano: dai tag, e in ripiego dal nome del file o della
cartella.

Nessun modello lo misura — l'anno non si sente, si legge — e la mappa fatta
finora non lo portava. Serve a Crate Talk: «anni 80» è un intervallo di anni,
e senza il numero sul brano resterebbe una parola.

Due sorgenti, in quest'ordine:

- i **tag**: `date` e `originaldate` come li espone mutagen in modalità
  easy, uguali su ID3, Vorbis comment e MP4. L'originale prima della
  data: una ristampa del 2011 di un disco del 1983 è musica del 1983, che
  è quello che il DJ chiede;
- il **nome**: un anno fra parentesi o quadre nel nome del file o della
  cartella — «(1983)», «[1985]» — o una cartella che comincia con l'anno,
  «1983 - Thriller». Un quattro cifre sciolto nel titolo NON vale:
  «Disco 2000» e «1999» sono titoli, non date, ed è meglio nessun anno di
  uno sbagliato che poi filtra una playlist.

Il backfill è quello di `titles`: si aggiunge il campo alle righe che non
lo hanno, senza toccare gli embedding (`MapStore.rewrite`). Manca il
CAMPO, non il valore: un brano senza anno scrive `None` e non si rilegge
a ogni giro.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # evita il giro map_profile → qui → map_store
    from .map_store import MapStore

# Gli anni che un brano può portare: fuori da qui è un numero, non una data.
FIRST_YEAR, LAST_YEAR = 1900, 2100

# Ogni quanti brani si salva: un backfill interrotto a metà riparte da dove
# era, e a novantamila brani riscrivere le righe costa un secondo.
FLUSH_EVERY = 1000

_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_BRACKETED = re.compile(r"[\(\[](\d{4})[\)\]]")
_LEADING = re.compile(r"^(\d{4})\b")


def _plausible(text: str | None) -> int | None:
    """Il primo gruppo di quattro cifre che sia un anno, o niente."""
    if not text:
        return None
    for found in _FOUR_DIGITS.findall(str(text)):
        year = int(found)
        if FIRST_YEAR <= year <= LAST_YEAR:
            return year
    return None


def year_from_tags(path: Path | str) -> int | None:
    """L'anno dai tag: l'originale se c'è, altrimenti la data. `None` se il
    file non si apre, non ha tag, o li ha senza un anno leggibile."""
    import mutagen

    try:
        audio = mutagen.File(path, easy=True)
    except Exception:                                   # noqa: BLE001
        return None
    if audio is None:
        return None
    for key in ("originaldate", "date"):
        values = audio.get(key) or []
        if isinstance(values, (str, bytes)):
            values = [values]
        for value in values:
            text = value.decode("utf-8", "ignore") \
                if isinstance(value, bytes) else str(value)
            year = _plausible(text)
            if year is not None:
                return year
    return None


def year_from_name(path: Path | str) -> int | None:
    """L'anno scritto nel nome del file o della cartella, fra parentesi o
    in testa alla cartella — e solo lì."""
    path = Path(path)
    for text in (path.stem, path.parent.name):
        for found in _BRACKETED.findall(text):
            year = _plausible(found)
            if year is not None:
                return year
    leading = _LEADING.match(path.parent.name)
    return _plausible(leading.group(1)) if leading else None


def year_of(path: Path | str) -> int | None:
    """L'anno di un brano, dai tag prima e dal nome poi."""
    year = year_from_tags(path)
    return year if year is not None else year_from_name(path)


def missing(rows: list[dict]) -> list[int]:
    """Le posizioni delle righe che l'anno non ha ancora visitato."""
    return [i for i, row in enumerate(rows) if "year" not in row]


def known(rows: list[dict]) -> int:
    """Quante righe portano un anno vero."""
    return sum(1 for row in rows if row.get("year") is not None)


def backfill(store: MapStore, flush_every: int = FLUSH_EVERY,
             on_progress=None) -> int:
    """Scrive l'anno sulle righe che non lo hanno. Ritorna quante.

    Un file che non c'è — disco staccato, brano spostato — si salta e resta
    da fare: scrivergli `None` lo segnerebbe per sempre come "senza anno".
    """
    todo = missing(store.rows)
    done = 0
    for i in todo:
        row = store.rows[i]
        if not os.path.exists(row["path"]):
            continue
        row["year"] = year_of(row["path"])
        done += 1
        if done % flush_every == 0:
            store.rewrite()
        if on_progress:
            on_progress(done, len(todo))
    if done:
        store.rewrite()
    return done
