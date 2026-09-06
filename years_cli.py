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

La chiave è la stessa di Describe (portachiavi di sistema, o la variabile
ANTHROPIC_API_KEY). Quello che parte per ogni brano: titolo e artista dai
tag, nome del file e della cartella. Niente altro.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis import api_keys, year_guess
from core.analysis.map_store import MapStore, default_store_dir

# Stime di spesa, per il dry-run: token per brano in ingresso e in uscita
# (misurati a spanne su righe come `year_guess.line`), e i prezzi batch di
# Opus 5 al momento di scrivere. Un ordine di grandezza, non una fattura.
TOKENS_IN_PER_TRACK, TOKENS_OUT_PER_TRACK = 30, 18
BATCH_USD_PER_MTOK_IN, BATCH_USD_PER_MTOK_OUT = 2.5, 12.5


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


def _estimate(count: int) -> str:
    usd = (count * TOKENS_IN_PER_TRACK * BATCH_USD_PER_MTOK_IN
           + count * TOKENS_OUT_PER_TRACK * BATCH_USD_PER_MTOK_OUT) / 1e6
    return f"~${usd:.2f}"


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
    args = parser.parse_args()
    if not (args.dry_run or args.submit or args.status or args.collect):
        parser.error("dimmi cosa fare: --dry-run, --submit, --status o --collect")

    store = MapStore.load(args.store)
    rows = store.rows
    todo = year_guess.candidates(rows)
    guessed = sum(1 for row in rows if row.get("year_guess") is not None)
    print(f"Mappa: {len(rows):,} brani · con anno dai tag "
          f"{sum(1 for r in rows if r.get('year') is not None):,} · stimati "
          f"{guessed:,} · da chiedere {len(todo):,}")

    if args.dry_run:
        count = min(len(todo), args.limit) if args.limit else len(todo)
        print(f"Un --submit manderebbe {count:,} brani in "
              f"{len(year_guess.chunks(todo[:count], args.per_request)):,} "
              f"richieste, {_estimate(count)} con {args.model} in batch.")
        if todo:
            print("\nLe prime righe, come le vede il modello:")
            for n, i in enumerate(todo[:5]):
                print("  " + year_guess.line(n + 1, rows[i]))
        return

    if args.status or args.collect:
        lots = year_guess.pending_lots(args.store)
        if not lots:
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
