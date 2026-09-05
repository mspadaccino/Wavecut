"""Titolo e artista dai tag, sulle righe della mappa che non li hanno.

Una mappa fatta prima che il profilo li leggesse ha il solo nome del file,
e con una libreria rippata da compilation il nome del file dice "Track 08".
Rileggere i tag costa qualche millisecondo a brano, l'analisi audio otto
secondi: si aggiunge il campo alle righe che ci sono gia' invece di rifare
la mappa. Vedi `MapStore.rewrite` per come si scrive senza toccare gli
embedding.
"""

from __future__ import annotations

import os

from .map_profile import read_tag_title_artist
from .map_store import MapStore

# Ogni quanti brani si salva: un backfill interrotto a meta' riparte da dove
# era, e a novantamila brani riscrivere le righe costa un secondo.
FLUSH_EVERY = 1000


def missing(rows: list[dict]) -> list[int]:
    """Le posizioni delle righe che i tag non hanno ancora visitato.

    Manca il CAMPO, non il valore: un file senza tag scrive due stringhe
    vuote e non si rilegge a ogni giro.
    """
    return [i for i, row in enumerate(rows) if "title" not in row]


def backfill(store: MapStore, flush_every: int = FLUSH_EVERY,
             on_progress=None) -> int:
    """Scrive titolo e artista sulle righe che non li hanno. Ritorna quante.

    Un file che non c'e' — disco staccato, brano spostato — si salta e
    resta da fare: scrivergli due stringhe vuote lo segnerebbe per sempre
    come "senza tag".
    """
    todo = missing(store.rows)
    done = 0
    for i in todo:
        row = store.rows[i]
        if not os.path.exists(row["path"]):
            continue
        row["title"], row["artist"] = read_tag_title_artist(row["path"])
        done += 1
        if done % flush_every == 0:
            store.rewrite()
        if on_progress:
            on_progress(done, len(todo))
    if done:
        store.rewrite()
    return done
