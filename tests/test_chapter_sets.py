"""Le cinque playlist dell'arco pescate dalla libreria: chi entra in quale
capitolo, e in che ordine ci sta dentro."""

import numpy as np
import pandas as pd

from core.analysis import chapter_sets
from core.analysis.arc import CHAPTERS
from core.analysis.map_profile import EMBEDDING_DIM, TrackProfile
from core.analysis.map_store import MapStore
from core.analysis.mixing import TransitionCost


def library(n: int = 50) -> pd.DataFrame:
    """Una libreria finta stesa su tutte e quattro le misure: il brano k sta
    al rango k/(n-1) su ognuna, così ogni capitolo trova chi lo riempie."""
    place = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({
        "path": [f"/x/{k}.mp3" for k in range(n)],
        "bpm": 100.0 + 40.0 * place,
        "energy": place,
        "valence_rank": place,
        "danceability": place,
        "camelot": ["8A" if k % 2 else "8B" for k in range(n)],
        "duration": 300.0,
    })


def cost_of(frame: pd.DataFrame) -> TransitionCost:
    # Vettori a ventaglio: la distanza di suono cresce con l'angolo, quindi
    # l'ordine dentro un capitolo ha qualcosa da minimizzare.
    angles = np.radians(np.linspace(0, 90, len(frame)))
    vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    return TransitionCost(vectors, frame["bpm"].tolist(),
                          frame["camelot"].tolist())


# --- la scelta ---

def test_every_chapter_gets_the_size_it_asked_for():
    chosen = chapter_sets.pick(library(50), size=5)
    assert [len(group) for group in chosen] == [5] * len(CHAPTERS)


def test_a_track_belongs_to_one_chapter_only():
    chosen = chapter_sets.pick(library(50), size=5)
    picked = [i for group in chosen for i in group]
    assert len(picked) == len(set(picked))


def test_the_tracks_of_a_chapter_fit_it_better_than_the_others():
    frame = library(50)
    ranking = chapter_sets.costs(frame)
    chosen = chapter_sets.pick(frame, size=4)
    for n, group in enumerate(chosen):
        mine = ranking[group, n].mean()
        others = ranking[:, n].mean()
        assert mine < others


def test_a_short_library_is_shared_out_and_not_drained_by_the_first():
    # Cinque brani e dieci chiesti a testa: a servire i capitoli in fila,
    # Intro se li prenderebbe tutti e cinque e gli altri quattro resterebbero
    # vuoti. A giro, ognuno ne prende uno.
    chosen = chapter_sets.pick(library(5), size=10)
    assert sum(len(group) for group in chosen) == 5
    assert all(len(group) == 1 for group in chosen)


def test_a_track_without_all_four_measures_is_not_picked():
    frame = library(10)
    frame.loc[3, "danceability"] = None
    frame.loc[7, "bpm"] = np.nan
    picked = {i for group in chapter_sets.pick(frame, size=2) for i in group}
    assert 3 not in picked and 7 not in picked


def test_nothing_measured_means_no_playlists():
    frame = library(10)
    frame["energy"] = np.nan
    assert chapter_sets.pick(frame, size=3) == [[] for _ in CHAPTERS]
    assert not chapter_sets.measured(frame).any()


def test_an_empty_library_gives_five_empty_chapters():
    empty = pd.DataFrame(columns=["bpm", "energy", "valence_rank",
                                  "danceability"])
    assert chapter_sets.costs(empty).shape == (0, len(CHAPTERS))
    assert chapter_sets.pick(empty) == [[] for _ in CHAPTERS]


def test_the_same_library_gives_the_same_playlists():
    frame = library(40)
    assert chapter_sets.pick(frame, size=6) == chapter_sets.pick(frame, size=6)


def test_the_indices_are_the_labels_of_the_frame():
    frame = library(20)
    frame.index = frame.index + 100
    picked = [i for group in chapter_sets.pick(frame, size=2) for i in group]
    assert all(i >= 100 for i in picked)


# --- l'ordine ---

def test_each_chapter_comes_out_ordered_and_whole():
    frame = library(50)
    cost = cost_of(frame)
    chosen = chapter_sets.pick(frame, size=5)
    playlists = chapter_sets.ordered(cost, chosen)
    for group, playlist in zip(chosen, playlists):
        assert sorted(playlist) == sorted(group)


def test_a_chapter_starts_where_the_previous_one_left_off():
    frame = library(50)
    cost = cost_of(frame)
    chosen = chapter_sets.pick(frame, size=5)
    playlists = chapter_sets.ordered(cost, chosen)
    for before, after in zip(playlists, playlists[1:]):
        if not before or len(after) < 2:
            continue
        joint = cost.between(before[-1], after[0])
        assert joint <= min(cost.between(before[-1], i) for i in after) + 1e-6


def test_build_is_the_choice_and_the_order_together():
    frame = library(30)
    cost = cost_of(frame)
    assert chapter_sets.build(frame, cost, size=3) == \
        chapter_sets.ordered(cost, chapter_sets.pick(frame, size=3))


# --- la libreria letta dalla mappa ---

def test_library_reads_every_row_and_derives_the_two_ranks(tmp_path):
    store = MapStore.load(tmp_path / "map")
    for k in range(3):
        audio = tmp_path / f"{k}.mp3"
        audio.write_bytes(b"x")
        store.append([TrackProfile(
            path=audio, duration=300.0, bpm=120.0 + k, camelot="8A",
            danceability=0.5,
            energy={"energy_density": 0.1 * k, "energy_bass": 0.2 * k,
                    "energy_bright": 0.3 * k, "energy_pulse": 0.4 * k},
            mood_numbers={"valence": -0.5 + 0.5 * k},
            embedding=np.full(EMBEDDING_DIM, float(k + 1), dtype=np.float32))])

    frame = chapter_sets.library(store)
    # Tutte e tre, senza aver mai proiettato: i capitoli non leggono la mappa.
    assert not store.projected and len(frame) == 3
    assert list(frame["energy"]) == [0.0, 0.5, 1.0]
    assert list(frame["valence_rank"]) == [0.0, 0.5, 1.0]
    assert chapter_sets.measured(frame).all()
