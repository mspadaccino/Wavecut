"""Una playlist per capitolo, pescata dalla libreria intera.

L'arco (`arc.py`) definisce cinque capitoli — Intro, Buildup, Tension,
Climax, Release — e finora lo leggevano due strumenti: il Chapter Builder,
che ripartisce nei cinque una playlist già fatta, e il Journey, che cerca i
brani per andare da qui a lì. Manca la domanda di chi prepara la serata
PRIMA della serata: dammi cento brani buoni per l'Intro, cento per il
Buildup, e così via — cinque scaffali da cui pescare quando si suona,
invece di una scaletta sola scritta a tavolino.

È quello che c'è qui, e sullo stesso arco degli altri due: le fasce sono le
loro, quindi un brano che il Chapter Builder metterebbe in Tension è uno che
di qua finisce nella playlist Tension. Cambiare le fasce cambia tutti e tre
insieme, che è la ragione per cui stanno scritte in un posto solo.

La scelta è di QUESTA libreria: le quattro misure sono ranghi su di lei — il
capitolo chiede "il 15% più lento", non un BPM — quindi le stesse cinque
playlist su una libreria diversa portano brani diversi, ed è giusto così.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.analysis import energy, mood_scale
from core.analysis.arc import CHAPTERS, arc_costs, measures
from core.analysis.map_store import MapStore
from core.analysis.mixing import TransitionCost, sorted_after

# Cento brani per capitolo: una playlist da cui scegliere per tutta la
# durata del capitolo, non una scaletta da suonare intera.
DEFAULT_SIZE = 100

# Le quattro colonne del frame su cui si legge un capitolo, nell'ordine in
# cui `arc.measures` le vuole.
COLUMNS = ("bpm", "energy", "valence_rank", "danceability")


def _numbers(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Una colonna come numeri, `nan` dove non c'è — colonna compresa.

    Una mappa fatta prima che una misura si registrasse non porta la sua
    colonna: meglio un capitolo che non trova nessuno di un errore a metà
    lavoro.
    """
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def library(store: MapStore) -> pd.DataFrame:
    """Le righe della mappa con le due misure derivate accanto alle altre.

    TUTTE le righe, non solo quelle piazzate. Un capitolo si legge su tempo,
    energia, valence e groove, e nessuna delle quattro passa dalla proiezione
    UMAP: chiedere anche le coordinate terrebbe fuori dalla serata i brani
    analizzati dopo l'ultimo ricalcolo, per un motivo che con la serata non
    c'entra. Le righe stanno in fila come gli embedding, quindi l'indice del
    frame è la posizione del brano nella libreria — quella con cui si
    interroga il costo di transizione.

    Energia e valence sono ranghi sulla libreria intera, come dappertutto:
    si calcolano qui una volta e non per brano.
    """
    rows = store.rows
    frame = pd.DataFrame(rows) if rows else pd.DataFrame()
    frame["energy"] = energy.from_rows(rows)
    frame["valence_rank"] = energy.ranks(
        np.asarray(mood_scale.from_rows(rows), dtype=float))
    return frame


def measured(frame: pd.DataFrame) -> np.ndarray:
    """Quali righe hanno davvero tutte e quattro le misure.

    `arc.measures` mette a metà scala quello che manca, e a metà scala un
    brano cade dentro alla fascia di qualcuno senza che nessuno lo abbia
    misurato. Per DISEGNARE un brano al centro va bene; per SCEGLIERLO no —
    si prenderebbe il posto di uno misurato.
    """
    ok = np.ones(len(frame), dtype=bool)
    for column in COLUMNS:
        ok &= np.isfinite(_numbers(frame, column))
    return ok


def costs(frame: pd.DataFrame) -> np.ndarray:
    """Quanto ogni brano sta fuori da ogni capitolo: N × 5, fra 0 e 1.

    Zero vuol dire dentro tutte e quattro le fasce di quel capitolo. È la
    stessa misura che insegue il Journey, sulla stessa scala.
    """
    values = measures(_numbers(frame, "bpm"), _numbers(frame, "energy"),
                      _numbers(frame, "valence_rank"),
                      _numbers(frame, "danceability"))
    if not len(values):
        return np.empty((0, len(CHAPTERS)), dtype=np.float32)
    return np.column_stack([arc_costs(values, n)
                            for n in range(len(CHAPTERS))])


def pick(frame: pd.DataFrame, size: int = DEFAULT_SIZE) -> list[list[int]]:
    """Per ogni capitolo, i `size` brani che stanno più dentro le sue fasce.

    Un brano va in un capitolo solo, e i capitoli scelgono A GIRO: uno alla
    volta prendono il migliore che gli è rimasto. Serviti in fila, il primo
    si porterebbe via tutti i brani contesi e l'ultimo avrebbe gli avanzi —
    Release fatto degli scarti dell'Intro, che sulle fasce di quei due
    capitoli è proprio il caso che capita.

    Se i brani dentro le fasce non bastano a riempire un capitolo, si
    continua con i più vicini a entrarci: cento brani si sono chiesti, cento
    se ne danno. Quanti stiano davvero dentro lo dice `costs`.

    Torna le etichette dell'indice del frame — le posizioni nella libreria,
    come le vuole ogni altra parte dell'app.
    """
    ranking = costs(frame)
    ok = measured(frame)
    labels = frame.index.to_numpy()

    # La fila d'attesa di ogni capitolo: le righe misurate, dalla più dentro
    # alla più fuori. Il pareggio si scioglie sull'ordine della libreria
    # (`stable`), così due lanci sulla stessa mappa danno le stesse playlist.
    queues = [[int(p) for p in np.argsort(ranking[:, n], kind="stable")
               if ok[p]]
              for n in range(len(CHAPTERS))]

    chosen: list[list[int]] = [[] for _ in CHAPTERS]
    cursors = [0] * len(CHAPTERS)
    taken: set[int] = set()
    while True:
        served = False
        for n, queue in enumerate(queues):
            if len(chosen[n]) >= size:
                continue
            k = cursors[n]
            while k < len(queue) and queue[k] in taken:
                k += 1
            if k >= len(queue):
                cursors[n] = k
                continue
            taken.add(queue[k])
            chosen[n].append(int(labels[queue[k]]))
            cursors[n] = k + 1
            served = True
        if not served:
            break
    return chosen


def ordered(cost: TransitionCost,
            chosen: list[list[int]]) -> list[list[int]]:
    """Ogni capitolo messo in fila, e agganciato a quello prima.

    Dentro il capitolo l'ordine è il cammino che costa meno — il Magic sort
    della pagina Map, la stessa funzione. Fra un capitolo e il successivo si
    parte dal brano che costa meno raggiungere dall'ultimo del precedente:
    le cinque playlist non sono cinque serate ma i cinque tratti di una, e
    la giuntura è il punto in cui si sentirebbe.
    """
    out: list[list[int]] = []
    for group in chosen:
        if not group:
            out.append([])
            continue
        previous = next((done for done in reversed(out) if done), [])
        out.append(sorted_after(cost, previous, group))
    return out


def build(frame: pd.DataFrame, cost: TransitionCost,
          size: int = DEFAULT_SIZE) -> list[list[int]]:
    """Le cinque playlist pronte: scelte per capitolo e messe in fila."""
    return ordered(cost, pick(frame, size))
