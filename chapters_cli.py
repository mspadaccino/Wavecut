#!/usr/bin/env python3
"""Entry point 5 — cinque playlist, una per capitolo dell'arco.

Setaccia la mappa capitolo per capitolo e scrive cinque scalette: Intro,
Buildup, Tension, Climax, Release, cento brani ciascuna. Non è una serata
già scritta — è il materiale con cui si suona una serata, uno scaffale per
ogni momento, da cui pescare quando ci si è dentro.

    poetry run python chapters_cli.py                    # 100 per capitolo
    poetry run python chapters_cli.py --size 60
    poetry run python chapters_cli.py --dry-run          # dice cosa farebbe
    poetry run python chapters_cli.py --out ~/Desktop/serata

Senza `--out` scrive nello scaffale (`~/Documents/DjCaddy/Playlists`): le
cinque playlist si trovano nella scheda Shelf dell'app, senza importare
niente. I capitoli sono numerati (`1 Intro`, `2 Buildup`, …) perché lo
scaffale li mette in ordine alfabetico e la serata ha il suo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.analysis import chapter_sets
from core.analysis.arc import CHAPTERS
from core.analysis.map_store import MapStore, default_store_dir
from core.analysis.mixing import TransitionCost
from core.analysis.shelf import Shelf


def _clock(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 \
        else f"{minutes} min"


def _top_genres(frame: pd.DataFrame, playlist: list[int], how_many: int = 3) -> str:
    if "top_genre" not in frame:
        return ""
    counted = frame.loc[playlist, "top_genre"].fillna("—").value_counts()
    return ", ".join(str(name) for name in counted.index[:how_many])


def _line(chapter: dict, frame: pd.DataFrame, playlist: list[int],
          inside: int) -> str:
    """Una riga per capitolo: quanti, quanto dura, e come suona."""
    if not playlist:
        return f"{chapter['icon']} {chapter['name']:<8} nessun brano"
    bpm = pd.to_numeric(frame.loc[playlist, "bpm"], errors="coerce").dropna()
    duration = pd.to_numeric(frame.loc[playlist, "duration"],
                             errors="coerce").fillna(0).sum() \
        if "duration" in frame else 0.0
    drive = pd.to_numeric(frame.loc[playlist, "energy"],
                          errors="coerce").dropna()
    pieces = [f"{len(playlist):>4} brani",
              f"{inside:>4} dentro le fasce",
              _clock(float(duration))]
    if len(bpm):
        pieces.append(f"{bpm.min():.0f}–{bpm.max():.0f} BPM")
    if len(drive):
        pieces.append(f"energia {drive.median():.2f}")
    genres = _top_genres(frame, playlist)
    if genres:
        pieces.append(genres)
    return f"{chapter['icon']} {chapter['name']:<8} " + " · ".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cinque playlist, una per capitolo dell'arco: Intro, "
                    "Buildup, Tension, Climax, Release.")
    parser.add_argument("--size", type=int, default=chapter_sets.DEFAULT_SIZE,
                        help="Quanti brani per capitolo (default: "
                             f"{chapter_sets.DEFAULT_SIZE})")
    parser.add_argument("--store", type=Path, default=default_store_dir(),
                        help="Cartella della mappa")
    parser.add_argument("--out", type=Path, default=None,
                        help="Dove scrivere le playlist (default: lo "
                             "scaffale che legge l'app)")
    parser.add_argument("--prefix", default="",
                        help="Davanti al nome di ogni playlist, per non "
                             "coprire quelle che ci sono già")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dice cosa scriverebbe, senza scrivere")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size vuole un numero di brani maggiore di zero")

    store = MapStore.load(args.store)
    frame = chapter_sets.library(store)
    if not len(frame):
        print(f"La mappa in {args.store} è vuota: costruiscila prima con "
              "map_cli.py.")
        return

    ok = chapter_sets.measured(frame)
    print(f"Mappa: {len(frame):,} brani · con tutte e quattro le misure "
          f"{int(ok.sum()):,}")
    if not ok.any():
        print("Nessun brano ha tempo, energia, valence e groove insieme: i "
              "capitoli si leggono su quelle quattro. Passa prima "
              "energy_cli.py (energia e groove) e mood_cli.py (valence).")
        return

    camelot = frame["camelot"].tolist() if "camelot" in frame \
        else [None] * len(frame)
    cost = TransitionCost(store.embeddings[:len(frame)],
                          frame["bpm"].tolist(), camelot)

    ranking = chapter_sets.costs(frame)
    chosen = chapter_sets.pick(frame, args.size)
    playlists = chapter_sets.ordered(cost, chosen)

    print()
    for n, (chapter, playlist) in enumerate(zip(CHAPTERS, playlists)):
        inside = int(np.sum(ranking[playlist, n] <= 0)) if playlist else 0
        print(_line(chapter, frame, playlist, inside))

    names = [f"{args.prefix}{n} {chapter['name']}"
             for n, chapter in enumerate(CHAPTERS, 1)]
    shelf = Shelf(args.out) if args.out else Shelf()
    if args.dry_run:
        print(f"\n--dry-run: in {shelf.folder} sarebbero andate "
              + ", ".join(f"«{name}»" for name in names) + ".")
        return

    written = []
    for name, playlist in zip(names, playlists):
        if not playlist:
            continue                      # un file vuoto non è una playlist
        shelf.write(name, [str(frame.at[i, "path"]) for i in playlist])
        written.append(name)
    print(f"\nScritte in {shelf.folder}: "
          + ", ".join(f"«{name}»" for name in written) + ".")


if __name__ == "__main__":
    sys.exit(main())
