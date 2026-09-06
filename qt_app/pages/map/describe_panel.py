"""La scheda Crate Buddy: una frase, e la playlist che le somiglia.

«Synth pop anni 80, solo versioni extended» si scrive nella casella; la
frase si LEGGE in un criterio — anni, generi, mood, tempo, parole nel
titolo, durata — e il criterio si mostra per intero PRIMA di cercare, in
XML: quello che non è scritto lì non si applica. Solo dopo si cerca, in
locale, sulla mappa (`core.analysis.describe`), e la lista va nella
playlist o sullo scaffale col nome della frase.

Il criterio si LEGGE e non si tocca. C'erano dei campi al suo posto, che
si correggevano a mano, ma li leggeva solo il bottone Search: chi spuntava
un genere e guardava la tabella vedeva una lista che quel genere non lo
aveva mai sentito. Meglio una cosa sola, vera: si corregge la frase e si
rilegge.

Chi legge la frase è in due. Il lettore a regole (`describe_lexicon`) c'è
sempre. Quello a modello (`describe_llm`) c'è con la chiave API
dell'utente: manda la frase e il vocabolario della libreria, non la
libreria, e se non risponde — niente rete, chiave sbagliata, credito
finito — si passa alle regole e lo si dice in una riga, senza finestre.
Le letture fatte si tengono su disco: la stessa frase non si ripaga. Le
nove collezioni in menu le legge il lessico, che le conosce per nome.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core.analysis import api_keys, describe, describe_lexicon
from core.analysis.describe import Query, Vocabulary
from core.analysis.describe_llm import (CANDIDATES_PER_PICK, ClaudeCurator,
                                        ClaudeReader, Curation, Readings,
                                        ReadingFailed)
from core.analysis.duplicates import song_key
from core.analysis.mixing import magic_sort
from core.analysis.shelf import Shelf, valid_name
from core.viz.track_columns import genre_colors
from qt_app import theme
from qt_app.widgets.track_table import TrackTable
from qt_app.workers import run_in_pool

from .library import Library
from .set_builder import numbered_rows

NO_COLLECTION = "— collections —"

# Cosa dice il criterio prima che una frase sia stata letta: si cerca lo
# stesso, e quello che esce è la libreria intera — meglio dirlo.
WAITING_FOR_A_PHRASE = ("<!-- no phrase read yet: searching now takes "
                        "the whole library -->")

# Dove si ricorda se chiedere a Claude: il modello costa credito, e chi ha
# la chiave deve poterlo tenere spento senza toglierla.
ASK_CLAUDE_KEY = "describe/ask_claude"
CURATE_KEY = "describe/curate"

# Le collezioni pronte: le frasi che il lessico conosce per nome. Si
# leggono senza chiave e senza rete, e servono anche da esempio di cosa
# scrivere nella casella.
COLLECTIONS = ["70s", "80s", "90s", "Flash House", "Ballads Remixes",
               "New Wave / Synth Pop", "ReVibes", "Rock", "Italo House",
               "Eurodance"]


def _dim(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("dim")
    label.setWordWrap(True)
    return label


def playlist_name(phrase: str) -> str:
    """Il nome di scaffale di una frase: la frase, se è un nome di file."""
    name = " ".join(phrase.split())[:60].strip().rstrip(".")
    return name if valid_name(name) else "Crate Buddy"


class DescribePanel(QWidget):
    """La casella, il criterio letto, la lista trovata."""

    append_playlist = Signal(list)          # gli INDICI di libreria spuntati
    shelve_playlist = Signal(str, list)     # nome sullo scaffale, indici

    def __init__(self, wire_table, shelf: Shelf | None = None,
                 readings: Readings | None = None, reader_factory=None,
                 keys=api_keys, settings: QSettings | None = None,
                 curator_factory=None, parent=None) -> None:
        super().__init__(parent)
        self._lib: Library | None = None
        self._vocabulary = Vocabulary()
        self._shelf = shelf or Shelf()
        self._readings = readings or Readings()
        self._settings = settings or QSettings(*theme.SETTINGS)
        # Come si costruisce il lettore a modello da una chiave: si passa
        # dai test, che un modello non lo chiamano.
        self._reader_factory = reader_factory or (
            lambda key: ClaudeReader(api_key=key))
        # Il curatore: chi sceglie dentro la rosa. Di default lo stesso
        # client del lettore; nei test, un finto.
        self._curator_factory = curator_factory or (
            lambda key: ClaudeCurator(api_key=key))
        self._keys = keys
        self._reading = False
        self._curating = False
        self._found: list[int] = []
        # Il criterio: quello che l'ultima lettura ha capito. Vuoto vuol
        # dire «tutto», ed è quello che si cerca premendo Search senza
        # aver letto niente.
        self._query = Query()
        self._read_by = ""
        self._build(wire_table)
        self._tell_reader()

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    def _build(self, wire_table) -> None:
        self._phrase = QLineEdit()
        self._phrase.setPlaceholderText(
            "💬 synth pop anni 80, solo versioni extended …")
        self._phrase.setClearButtonEnabled(True)
        self._phrase.returnPressed.connect(self._on_read)
        self._collections = QComboBox()
        self._collections.addItem(NO_COLLECTION)
        self._collections.addItems(COLLECTIONS)
        self._collections.setToolTip(theme.hint(
            "Ready-made phrases the rules know by name. Pick one and it "
            "is read at once, no key needed — and it shows what a phrase "
            "can say."))
        self._collections.currentTextChanged.connect(self._on_collection)
        self._read = QPushButton("🧠 Read")
        self._read.setToolTip(theme.hint(
            "Turns the phrase into the form below — years, genres, moods, "
            "tempo, words in the title, length — and shows it BEFORE "
            "searching, so a wrong reading is fixed here, not heard "
            "later. With an API key the phrase is read by Claude; without "
            "one, or when the network is down, by the rules. Only the "
            "phrase and the library's label vocabulary are sent: never "
            "the tracks."))
        self._read.clicked.connect(self._on_read)
        phrase_row = QHBoxLayout()
        phrase_row.addWidget(self._phrase, stretch=1)
        phrase_row.addWidget(self._collections)
        phrase_row.addWidget(self._read)

        self._reader_told = _dim("")
        self._ask_claude = QCheckBox("Ask Claude")
        self._ask_claude.setToolTip(theme.hint(
            "Whether a phrase is sent to Claude at all. Untick it and the "
            "rules read every phrase, key or no key, and nothing is spent: "
            "for a night without credit, or for phrases the rules already "
            "know. Remembered for next time."))
        self._ask_claude.setChecked(
            str(self._settings.value(ASK_CLAUDE_KEY, "true")).lower()
            != "false")
        self._ask_claude.toggled.connect(self._on_ask_claude)
        self._key = QPushButton("🔑 API key…")
        self._key.setToolTip(theme.hint(
            "Your Anthropic API key, kept in the system keychain. It "
            "unlocks reading the phrase with Claude — about a cent a "
            "phrase, on your own credit. Leave it empty to forget it."))
        self._key.clicked.connect(self._on_key)
        reader_row = QHBoxLayout()
        reader_row.addWidget(self._reader_told, stretch=1)
        reader_row.addWidget(self._ask_claude)
        reader_row.addWidget(self._key)

        # --- il criterio, com'è stato letto ---
        # Si LEGGE, non si tocca. C'erano dei campi al suo posto — anni,
        # generi, mood, tempo — che si potevano correggere a mano, ma
        # contavano solo alla pressione di Search: chi spuntava un genere
        # e guardava la tabella vedeva una lista che quel genere non lo
        # aveva mai sentito. Un campo che si lascia toccare e non conta è
        # una bugia; una frase da riscrivere no. Si corregge la frase e si
        # rilegge, che è poi il gesto di tutta la scheda.
        self._criterion = QLabel(WAITING_FOR_A_PHRASE)
        self._criterion.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._criterion.setWordWrap(True)
        self._criterion.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._criterion.setToolTip(theme.hint(
            "The whole of what is searched, and nothing else: what is not "
            "written here does not apply. <b>Filters</b> say who can "
            "enter — years, tempo, title words, length. <b>Seeds</b> say "
            "who enters first: the tracks carrying those labels, with the "
            "rest of the list filled by what sounds like them. To change "
            "it, change the phrase and read it again."))

        self._years_hint = _dim("")

        # --- la ricerca ---
        self._size = QSpinBox()
        self._size.setRange(1, 1000)
        self._size.setValue(describe.DEFAULT_SIZE)
        self._size.setToolTip(theme.hint(
            "How many tracks to bring back: a shelf to play a night "
            "from, not a set to play whole."))
        self._variety = QDoubleSpinBox()
        self._variety.setRange(0.0, 1.0)
        self._variety.setSingleStep(0.1)
        self._variety.setValue(describe.DEFAULT_VARIETY)
        self._variety.setToolTip(theme.hint(
            "<b>0 = close, doubles allowed · 1 = spread out</b><br>How far "
            "the fill-up wanders from the seeds — the Radio Mix knob."))
        self._curate = QCheckBox("Curate with Claude")
        self._curate.setToolTip(theme.hint(
            "After the local search, the shortlist — three candidates for "
            "every track wanted, one line each: title, artist, year, "
            "tempo, key, labels — goes to Claude with your phrase, and "
            "Claude keeps the best ones in its order, with a reason for "
            "the first few. It knows the records: which are the classics, "
            "which versions DJs play. A few cents a search, on your "
            "credit. Needs the key and «Ask Claude»."))
        self._curate.setChecked(
            str(self._settings.value(CURATE_KEY, "false")).lower() == "true")
        self._curate.toggled.connect(
            lambda on: self._settings.setValue(CURATE_KEY, "true" if on else "false"))
        self._search = QPushButton("🔎 Search")
        self._search.setToolTip(theme.hint(
            "Applies the form to the map, in this app, with nothing sent "
            "anywhere: the hard filters (years, tempo, length, title) say "
            "who can enter, the labels say who enters first, and the Radio "
            "Mix fills up to the count with what sounds like them."))
        self._search.clicked.connect(self._on_search)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Tracks"))
        search_row.addWidget(self._size)
        search_row.addSpacing(8)
        search_row.addWidget(QLabel("Variety"))
        search_row.addWidget(self._variety)
        search_row.addSpacing(8)
        search_row.addWidget(self._curate)
        search_row.addStretch(1)
        search_row.addWidget(self._search)

        self._found_told = _dim("")
        self._reasons = _dim("")
        self._reasons.setVisible(False)
        self._table = TrackTable(checkable=True, favouritable=True)
        wire_table(self._table)
        self._table.setVisible(False)
        pick_all = QPushButton("Select all")
        pick_all.clicked.connect(lambda: self._table.set_all_picked(True))
        pick_none = QPushButton("Select none")
        pick_none.clicked.connect(lambda: self._table.set_all_picked(False))
        pick_row = QHBoxLayout()
        pick_row.addWidget(pick_all)
        pick_row.addWidget(pick_none)
        pick_row.addStretch(1)
        self._add = QPushButton("➕ Add selected to the playlist")
        self._add.clicked.connect(self._on_add)
        self._shelve = QPushButton("📚 Save as a playlist named after the phrase")
        self._shelve.setToolTip(theme.hint(
            "Writes the ticked tracks to the shelf as a playlist named "
            "after the phrase, and brings it onto the table."))
        self._shelve.clicked.connect(self._on_shelve)
        self._add.setEnabled(False)
        self._shelve.setEnabled(False)
        send_row = QHBoxLayout()
        send_row.addWidget(self._add)
        send_row.addWidget(self._shelve)

        panel = QWidget()
        box = QVBoxLayout(panel)
        box.addLayout(phrase_row)
        box.addLayout(reader_row)
        box.addWidget(self._criterion)
        box.addWidget(self._years_hint)
        box.addLayout(search_row)
        box.addWidget(self._found_told)
        box.addWidget(self._reasons)
        box.addLayout(pick_row)
        box.addWidget(self._table, stretch=2)
        box.addLayout(send_row)

        # Nessuna cornice che scorre qui dentro: la scheda sta in Set
        # Curator, che scorre già per tutte le sue — due barre annidate
        # sono una barra di troppo.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    # ------------------------------------------------------------------
    # la libreria
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        self._lib = lib
        frame = lib.frame
        self._vocabulary = Vocabulary.of(frame)
        dated = int(describe.years_of(frame).notna().sum())
        guessed = int(describe.guessed_years(frame).sum())
        self._years_hint.setText(
            f"{dated:,} of {len(frame):,} tracks carry a year"
            + (f" ({guessed:,} estimated by Claude)" if guessed else "")
            + ("" if dated == len(frame) else
               " — the rest can be estimated: years_cli.py"))
        self._years_hint.setToolTip(theme.hint(
            "A filter on years keeps only tracks that carry one: a track "
            "without a year is not an 80s track, it is an unknown one. "
            "Years are read when a track goes on the map, from its tags or "
            "from a year in brackets in the file or folder name. For the "
            "tracks that have neither, <code>years_cli.py</code> asks "
            "Claude the original release year, once, in batch; an estimate "
            "counts only when Claude is fairly sure of it."))

    # ------------------------------------------------------------------
    # chi legge
    # ------------------------------------------------------------------
    def asks_claude(self) -> bool:
        """Se una frase va a Claude: con una chiave, e con la spunta."""
        return self._ask_claude.isChecked() and bool(self._keys.read())

    def _on_ask_claude(self, on: bool) -> None:
        self._settings.setValue(ASK_CLAUDE_KEY, "true" if on else "false")
        self._tell_reader()

    def _tell_reader(self, trouble: str | None = None) -> None:
        key = self._keys.read()
        if trouble:
            self._reader_told.setText(f"⚠️ {trouble} Read by the rules.")
        elif key and not self._ask_claude.isChecked():
            self._reader_told.setText(
                "Phrases are read by the rules: Claude is off, nothing is "
                "spent. Tick «Ask Claude» to use your key.")
        elif key:
            self._reader_told.setText(
                f"Phrases are read by Claude · key in {self._keys.where()}")
        else:
            self._reader_told.setText(
                "Phrases are read by the rules. Add your API key under 🔑 to "
                "have Claude read them.")

    def _on_key(self) -> None:
        current = self._keys.read() or ""
        text, ok = QInputDialog.getText(
            self, "Anthropic API key",
            "Your API key, from console.anthropic.com. Only the phrase and "
            "the library's label names are ever sent.\nLeave empty to "
            "forget the key.",
            QLineEdit.EchoMode.Password, current)
        if not ok:
            return
        where = self._keys.write(text)
        self._tell_reader()
        if text.strip():
            QMessageBox.information(self, "API key", f"Kept in {where}.")

    def _on_collection(self, name: str) -> None:
        if name == NO_COLLECTION:
            return
        self._phrase.setText(name)
        self._show(describe_lexicon.read(name, self._vocabulary), "the rules")

    def _on_read(self) -> None:
        text = self._phrase.text().strip()
        if not text or self._reading:
            return
        self._collections.blockSignals(True)
        self._collections.setCurrentIndex(0)
        self._collections.blockSignals(False)
        remembered = self._readings.get(text)
        if remembered is not None:
            self._show(remembered.cleaned(self._vocabulary), "memory")
            return
        if not self.asks_claude():
            self._show(describe_lexicon.read(text, self._vocabulary),
                       "the rules")
            return
        key = self._keys.read()
        self._reading = True
        self._read.setEnabled(False)
        self._criterion.setText("<!-- reading… -->")
        reader = self._reader_factory(key)
        vocabulary = self._vocabulary
        run_in_pool(lambda: reader.read(text, vocabulary),
                    lambda query: self._on_model_read(text, query),
                    lambda trouble: self._on_model_failed(text, trouble))

    def _on_model_read(self, text: str, query: Query) -> None:
        self._reading = False
        self._read.setEnabled(True)
        self._readings.put(text, query)
        self._tell_reader()
        self._show(query, "Claude")

    def _on_model_failed(self, text: str, trouble: Exception) -> None:
        self._reading = False
        self._read.setEnabled(True)
        line = str(trouble) if isinstance(trouble, ReadingFailed) \
            else f"The model could not be reached ({type(trouble).__name__})."
        self._tell_reader(line)
        self._show(describe_lexicon.read(text, self._vocabulary), "the rules")

    # ------------------------------------------------------------------
    # il criterio
    # ------------------------------------------------------------------
    def _show(self, query: Query, by: str) -> None:
        """La lettura diventa IL criterio: da qui in poi è questo che si
        cerca, finché non se ne legge un altro."""
        self._query = query
        self._read_by = by
        self._criterion.setText(
            describe.as_xml(query, phrase=self._phrase.text().strip(),
                            read_by=by))

    def query(self) -> Query:
        """Il criterio letto per ultimo: è questo che si cerca."""
        query = self._query
        if not query.how_read:
            query.how_read = describe.summary(query)
        return query

    # ------------------------------------------------------------------
    # la ricerca
    # ------------------------------------------------------------------
    def curates(self) -> bool:
        return self._curate.isChecked() and self.asks_claude()

    def _on_search(self) -> None:
        if self._lib is None or self._curating:
            return
        lib, frame = self._lib, self._lib.frame
        query = self.query()
        size = self._size.value()
        # Con la cura, la ricerca locale porta una rosa più larga e Claude
        # tiene i migliori; senza, porta la lista e basta.
        wanted = size * CANDIDATES_PER_PICK if self.curates() else size
        found = describe.search(
            frame, lib.store.embeddings[:len(frame)], query,
            size=wanted, variety=self._variety.value(),
            song_of=lambda i: song_key(Path(frame.at[i, "path"])))
        if not self.curates() or not found.tracks:
            self._show_found(found, found.tracks[:size], {}, "")
            return
        self._curating = True
        self._search.setEnabled(False)
        self._found_told.setText(
            f"Claude is choosing {size} out of {len(found.tracks)}…")
        curator = self._curator_factory(self._keys.read())
        phrase = self._phrase.text()
        run_in_pool(lambda: curator.curate(phrase, query, frame,
                                           found.tracks, size),
                    lambda curation: self._on_curated(found, size, curation),
                    lambda trouble: self._on_curation_failed(found, size, trouble))

    def _on_curated(self, found: describe.Match, size: int,
                    curation: Curation) -> None:
        self._curating = False
        self._search.setEnabled(True)
        picks = curation.picks or found.tracks[:size]
        told = (f"Claude kept {len(curation.picks)} of {len(found.tracks)}"
                if curation.picks else
                "Claude kept none — showing the local list")
        self._show_found(found, picks, curation.reasons, told)

    def _on_curation_failed(self, found: describe.Match, size: int,
                            trouble: Exception) -> None:
        self._curating = False
        self._search.setEnabled(True)
        line = str(trouble) if isinstance(trouble, ReadingFailed) \
            else f"The model could not be reached ({type(trouble).__name__})."
        self._show_found(found, found.tracks[:size], {},
                         f"⚠️ {line} Showing the local list.")

    def _show_found(self, found: describe.Match, tracks: list[int],
                    reasons: dict[int, str], curated: str) -> None:
        lib, frame = self._lib, self._lib.frame
        ordered = magic_sort(lib.cost, tracks)
        self._found = ordered
        shown = numbered_rows(frame, ordered, lib.common)
        self._table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
        self._table.set_all_picked(True)
        self._table.setVisible(bool(ordered))
        self._add.setEnabled(bool(ordered))
        self._shelve.setEnabled(bool(ordered))
        self._found_told.setText(
            self._tell_found(found, len(tracks), len(frame))
            + (f" · {curated}" if curated else ""))
        lines = [f"<b>{frame.at[i, 'title'] or frame.at[i, 'name']}</b> — {why}"
                 for i, why in reasons.items() if i in ordered]
        self._reasons.setText("<br>".join(lines))
        self._reasons.setVisible(bool(lines))

    @staticmethod
    def _tell_found(found: describe.Match, shown: int, total: int) -> str:
        if not shown:
            told = "No track matches"
            if found.no_year and found.no_year == total:
                told += " — no track on the map carries a year yet"
            return told + "."
        pieces = [f"{shown} track(s)",
                  f"{len(found.pool):,} of {total:,} pass the filters"]
        if found.seeds:
            pieces.append(f"{len(found.seeds)} carry the labels")
        if found.guessed:
            pieces.append(f"{found.guessed} dated by Claude's estimate")
        if found.no_year:
            pieces.append(f"{found.no_year:,} without a year left out")
        return " · ".join(pieces)

    def _picked(self) -> list[int]:
        at_path = self._lib.at_path
        return [at_path[p] for p in self._table.selected_paths()
                if p in at_path]

    def _on_add(self) -> None:
        wanted = self._picked()
        if wanted:
            self.append_playlist.emit(wanted)

    def _on_shelve(self) -> None:
        wanted = self._picked()
        if not wanted:
            return
        name = playlist_name(self._phrase.text() or "Crate Buddy")
        if name in self._shelf.names():
            answer = QMessageBox.question(
                self, "Save the playlist", f"Overwrite «{name}» on the shelf?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.shelve_playlist.emit(name, wanted)
