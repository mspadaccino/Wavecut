#!/usr/bin/env python3
"""Entry point 5 — l'anno stimato da Claude per i brani che i tag non datano.

Chiede a Claude, via Batch API (metà prezzo), l'anno di uscita ORIGINALE di
ogni brano senza anno, e lo scrive in `year_guess` con la sua fiducia. Il
tag `year`, quando c'è, non si tocca. Vedi `core/analysis/year_guess.py`.

    poetry run python years_cli.py --dry-run          # quanti, e cosa costa
    poetry run python years_cli.py --submit --limit 200   # una prova
    poetry run python years_cli.py --submit           # tutto il resto
    poetry run python years_cli.py --status           # a che punto è
    poetry run python years_cli.py --collect          # scrive le risposte

Un lotto dura di solito meno di un'ora. Fra `--submit` e `--collect` si
può chiudere tutto: il lotto è segnato su disco accanto alla mappa.

La via senza API, con la chat di Claude che sta nell'abbonamento:

    poetry run python years_cli.py --export ~/Desktop/anni    # i file .txt
    # ogni file va dato alla chat («compila questo»), la risposta salvata
    # accanto con lo stesso nome e la coda «-answer.txt»
    poetry run python years_cli.py --import ~/Desktop/anni/years-001-answer.txt

Manuale, ma gratis. L'app non guida la chat da sola: non si fa.

La chiave è la stessa di Crate Talk (portachiavi di sistema, o la variabile
ANTHROPIC_API_KEY). Quello che parte per ogni brano: titolo e artista dai
tag, nome del file e della cartella. Niente altro.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis import api_keys, year_guess
from core.analysis.map_store import MapStore, default_store_dir

# Stime di spesa, per il dry-run: token per brano in ingresso e in uscita,
# TARATI su un lotto vero (42.000 brani, Sonnet 5 in batch, circa dieci
# dollari): il prologo si paga a ogni richiesta, e una riga da DJ è lunga.
# Poi i prezzi batch — metà del listino — al momento di scrivere, per
# milione di token, in e out. Un modello non in tabella si stima come
# Opus. Un ordine di grandezza, non una fattura.
TOKENS_IN_PER_TRACK, TOKENS_OUT_PER_TRACK = 75, 30
BATCH_USD_PER_MTOK = {
    "claude-opus-5": (2.5, 12.5),
    "claude-sonnet-5": (1.0, 5.0),
    "claude-haiku-4-5": (0.5, 2.5),
}


def _client():
    key = api_keys.read()
    if not key:
        sys.exit("Nessuna chiave API: mettila nell'app sotto 🔑, o esporta "
                 f"{api_keys.ENV_VAR}.")
    try:
        import anthropic
    except ImportError:
        sys.exit("Manca il pacchetto `anthropic`: poetry install --with describe.")
    return anthropic.Anthropic(api_key=key)


def _estimate(count: int, model: str) -> str:
    usd_in, usd_out = BATCH_USD_PER_MTOK.get(model, BATCH_USD_PER_MTOK["claude-opus-5"])
    usd = (count * TOKENS_IN_PER_TRACK * usd_in
           + count * TOKENS_OUT_PER_TRACK * usd_out) / 1e6
    return f"~${usd:.2f}"


def _show(rows: list[dict], count: int) -> None:
    """Le stime a campione, e come si distribuisce la fiducia: è il modo
    di decidere se il lotto grande vale la spesa."""
    import random

    asked = [r for r in rows if "year_guess" in r]
    if not asked:
        print("Nessuna stima ancora: prima --submit, poi --collect.")
        return
    dated = [r for r in asked if r.get("year_guess") is not None]
    sure = [r for r in dated
            if (r.get("year_guess_conf") or 0) >= year_guess.MIN_CONFIDENCE]
    print(f"Chieste {len(asked):,} · con un anno {len(dated):,} · abbastanza "
          f"sicure (≥ {year_guess.MIN_CONFIDENCE}) {len(sure):,} — solo "
          "queste filtrano")
    print("\nA campione (fiducia · anno · brano):")
    for row in random.Random(0).sample(asked, min(count, len(asked))):
        year = row.get("year_guess")
        print(f"  {row.get('year_guess_conf') or 0:.2f} · "
              f"{year if year is not None else '----'} · "
              + year_guess.line(0, row)[3:])


def _import(store, store_dir: Path, files: list[Path]) -> None:
    """Le risposte della chat sulla mappa: il file dice a quale lotto e a
    quale pezzo appartiene, per nome."""
    lots = [lot for lot in year_guess.pending_lots(store_dir)
            if lot.model == "chat"]
    if not lots:
        sys.exit("Nessun export in attesa: prima --export DIR.")
    written = 0
    for path in files:
        name = path.stem.removesuffix("-answer")
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError as trouble:
            print(f"{path.name}: non si legge ({trouble})")
            continue
        for lot in lots:
            dated = year_guess.import_answer(store, lot, name, text)
            if dated >= 0:
                answered = len(year_guess.parse_chat_answer(text))
                asked = len(lot.requests[name])
                print(f"{path.name}: {answered:,} risposte su {asked:,} "
                      f"brani · {dated:,} datati")
                del lot.requests[name]
                year_guess.save_lot(store_dir, lot) if lot.requests \
                    else year_guess.forget_lot(store_dir, lot)
                written += answered
                break
        else:
            print(f"{path.name}: nessun export si chiama «{name}» — i file "
                  "vanno tenuti col loro nome")
    if written:
        store.rewrite()
    left = sum(lot.tracks for lot in year_guess.pending_lots(store_dir)
               if lot.model == "chat")
    print(f"Stimati ora: "
          f"{sum(1 for r in store.rows if r.get('year_guess') is not None):,}"
          + (f" · ancora da importare {left:,} brani" if left else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L'anno stimato da Claude per i brani senza anno.")
    parser.add_argument("--store", type=Path, default=default_store_dir())
    parser.add_argument("--model", default=year_guess.DEFAULT_MODEL)
    parser.add_argument("--per-request", type=int, default=year_guess.PER_REQUEST,
                        help="Brani per richiesta del lotto")
    parser.add_argument("--limit", type=int, default=0,
                        help="Manda solo i primi N brani (una prova)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quanti brani si chiederebbero, e a che costo")
    parser.add_argument("--submit", action="store_true",
                        help="Manda un lotto per i brani senza anno")
    parser.add_argument("--status", action="store_true",
                        help="I lotti in attesa, e a che punto sono")
    parser.add_argument("--collect", action="store_true",
                        help="Scrive sulla mappa le risposte dei lotti finiti")
    parser.add_argument("--export", type=Path, metavar="DIR",
                        help="Scrive i brani da chiedere in file .txt per la "
                             "chat di Claude, in DIR")
    parser.add_argument("--per-file", type=int, default=year_guess.PER_FILE,
                        help="Brani per file di testo")
    parser.add_argument("--import", dest="import_", type=Path, nargs="+",
                        metavar="FILE",
                        help="Rilegge le risposte della chat salvate in FILE "
                             "(years-001-answer.txt, o years-001.txt)")
    parser.add_argument("--show", type=int, metavar="N", default=0,
                        help="Mostra N stime a campione, con la fiducia, per "
                             "giudicarle a orecchio")
    args = parser.parse_args()
    if not (args.dry_run or args.submit or args.status or args.collect
            or args.show or args.export or args.import_):
        parser.error("dimmi cosa fare: --dry-run, --submit, --status, "
                     "--collect, --show N, --export DIR o --import FILE")

    store = MapStore.load(args.store)
    rows = store.rows
    todo = year_guess.candidates(rows)
    guessed = sum(1 for row in rows if row.get("year_guess") is not None)
    print(f"Mappa: {len(rows):,} brani · con anno dai tag "
          f"{sum(1 for r in rows if r.get('year') is not None):,} · stimati "
          f"{guessed:,} · da chiedere {len(todo):,}")

    if args.show:
        _show(rows, args.show)
        return

    if args.export:
        lot = year_guess.export(store, args.export, per_file=args.per_file,
                                limit=args.limit)
        if lot is None:
            print("Niente da chiedere.")
            return
        year_guess.save_lot(args.store, lot)
        print(f"Scritti {len(lot.requests):,} file in {args.export} per "
              f"{lot.tracks:,} brani. Ogni file va dato alla chat di Claude "
              "(«compila questo»); salva la risposta accanto, con lo stesso "
              "nome e la coda -answer.txt, poi `--import` su quei file.")
        return

    if args.import_:
        _import(store, args.store, args.import_)
        return

    if args.dry_run:
        count = min(len(todo), args.limit) if args.limit else len(todo)
        print(f"Un --submit manderebbe {count:,} brani in "
              f"{len(year_guess.chunks(todo[:count], args.per_request)):,} "
              f"richieste, {_estimate(count, args.model)} con {args.model} "
              "in batch.")
        if todo:
            print("\nLe prime righe, come le vede il modello:")
            for n, i in enumerate(todo[:5]):
                print("  " + year_guess.line(n + 1, rows[i]))
        return

    if args.status or args.collect:
        every = year_guess.pending_lots(args.store)
        for lot in every:
            if lot.model == "chat":
                print(f"{lot.batch_id}: export per la chat, {lot.tracks:,} "
                      f"brani in {len(lot.requests):,} file ancora da importare")
        lots = [lot for lot in every if lot.model != "chat"]
        if not every:
            print("Nessun lotto in attesa.")
        client = _client() if lots else None
        for lot in lots:
            if args.collect:
                got = year_guess.collect(client, store, lot)
                if got.ended:
                    print(f"{lot.batch_id}: finito · {got.answered:,} risposte "
                          f"· {got.dated:,} brani datati · {got.failed:,} "
                          "richieste senza risposta")
                    year_guess.forget_lot(args.store, lot)
                else:
                    print(f"{lot.batch_id}: in lavorazione, {got.processing:,} "
                          "richieste ancora da fare — riprova più tardi")
            else:
                batch = client.messages.batches.retrieve(lot.batch_id)
                counts = batch.request_counts
                print(f"{lot.batch_id}: {batch.processing_status} · "
                      f"{lot.tracks:,} brani in {len(lot.requests):,} richieste "
                      f"· fatte {counts.succeeded:,} · in corso "
                      f"{counts.processing:,} · errori {counts.errored:,}")
        if args.collect and lots:
            print(f"\nStimati ora: "
                  f"{sum(1 for r in rows if r.get('year_guess') is not None):,}")
        return

    if args.submit:
        if not todo:
            print("Niente da chiedere.")
            return
        lot = year_guess.submit(_client(), store, model=args.model,
                                per_request=args.per_request, limit=args.limit)
        year_guess.save_lot(args.store, lot)
        print(f"Mandato il lotto {lot.batch_id}: {lot.tracks:,} brani in "
              f"{len(lot.requests):,} richieste. Fra un po' "
              "`years_cli.py --status`, poi `--collect`.")


if __name__ == "__main__":
    main()
