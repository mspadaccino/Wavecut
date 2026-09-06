"""La scheda Describe: una frase, e la playlist che le somiglia.

«Synth pop anni 80, solo versioni extended» si scrive nella casella; la
frase si LEGGE in un modulo — anni, generi, mood, tempo, parole nel
titolo, durata — e il modulo si mostra PRIMA di cercare, in campi che si
correggono a mano. Solo dopo la conferma si cerca, in locale, sulla mappa
(`core.analysis.describe`), e la lista va nella playlist o sullo
scaffale col nome della frase.

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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core.analysis import api_keys, describe, describe_lexicon
from core.analysis.describe import Query, Vocabulary
from core.analysis.describe_llm import ClaudeReader, Readings, ReadingFailed
from core.analysis.duplicates import song_key
from core.analysis.mixing import magic_sort
from core.analysis.shelf import Shelf, valid_name
from core.analysis.years import FIRST_YEAR, LAST_YEAR
from core.viz.filters import span
from core.viz.track_columns import genre_colors
from qt_app import theme
from qt_app.pages.common import scrollable
from qt_app.widgets.range_slider import RangeSlider
from qt_app.widgets.track_table import TrackTable
from qt_app.workers import run_in_pool

from .filters import CheckList
from .library import Library
from .set_builder import numbered_rows

NO_COLLECTION = "— collections —"

# Dove si ricorda se chiedere a Claude: il modello costa credito, e chi ha
# la chiave deve poterlo tenere spento senza toglierla.
ASK_CLAUDE_KEY = "describe/ask_claude"

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
    return name if valid_name(name) else "Describe"


class DescribePanel(QWidget):
    """La casella, il modulo letto e correggibile, la lista trovata."""

    append_playlist = Signal(list)          # gli INDICI di libreria spuntati
    shelve_playlist = Signal(str, list)     # nome sullo scaffale, indici

    def __init__(self, wire_table, shelf: Shelf | None = None,
                 readings: Readings | None = None, reader_factory=None,
                 keys=api_keys, settings: QSettings | None = None,
                 parent=None) -> None:
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
        self._keys = keys
        self._reading = False
        self._found: list[int] = []
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

        # --- il modulo, com'è stato letto e come si corregge ---
        self._how_read = QLabel("")
        self._how_read.setWordWrap(True)
        self._how_read.setToolTip(theme.hint(
            "How the phrase was read, in one line. The fields below are "
            "that reading: change any of them and search."))

        self._years_on = QCheckBox("Years")
        self._year_from, self._year_to = QSpinBox(), QSpinBox()
        for spin in (self._year_from, self._year_to):
            spin.setRange(FIRST_YEAR, LAST_YEAR)
        self._year_from.setValue(1980)
        self._year_to.setValue(1989)
        self._years_on.toggled.connect(self._year_from.setEnabled)
        self._years_on.toggled.connect(self._year_to.setEnabled)
        self._years_on.setChecked(False)
        self._year_from.setEnabled(False)
        self._year_to.setEnabled(False)
        self._years_hint = _dim("")
        years_row = QHBoxLayout()
        years_row.addWidget(self._years_on)
        years_row.addWidget(self._year_from)
        years_row.addWidget(QLabel("–"))
        years_row.addWidget(self._year_to)
        years_row.addWidget(self._years_hint, stretch=1)

        self._genres = CheckList("Genres")
        self._genres.setToolTip(theme.hint(
            "The genre labels the phrase named. Tracks carrying them are "
            "the seeds: they enter first, strongest first, and the rest "
            "of the list is what sounds like them."))
        self._moods = CheckList("Moods")
        lists_row = QHBoxLayout()
        lists_row.addWidget(self._genres, stretch=1)
        lists_row.addWidget(self._moods, stretch=1)

        self._bpm_on = QCheckBox("BPM")
        self._bpm = RangeSlider(decimals=0)
        self._bpm.set_span(describe.BPM_FLOOR, describe.BPM_CEILING)
        self._bpm_on.toggled.connect(self._bpm.setEnabled)
        self._bpm_on.setChecked(False)
        self._bpm.setEnabled(False)
        self._title_words = QLineEdit()
        self._title_words.setPlaceholderText("words the title must have, "
                                             "comma-separated: remix, extended")
        self._title_words.setClearButtonEnabled(True)
        self._minutes_on = QCheckBox("At least")
        self._minutes = QDoubleSpinBox()
        self._minutes.setRange(0.5, 60.0)
        self._minutes.setSingleStep(0.5)
        self._minutes.setDecimals(1)
        self._minutes.setValue(5.5)
        self._minutes.setSuffix(" min")
        self._minutes_on.toggled.connect(self._minutes.setEnabled)
        self._minutes_on.setChecked(False)
        self._minutes.setEnabled(False)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._bpm_on, 0, 0)
        grid.addWidget(self._bpm, 0, 1)
        grid.addWidget(QLabel("Title"), 1, 0)
        grid.addWidget(self._title_words, 1, 1)
        minutes_row = QHBoxLayout()
        minutes_row.setContentsMargins(0, 0, 0, 0)
        minutes_row.addWidget(self._minutes)
        minutes_row.addStretch(1)
        grid.addWidget(self._minutes_on, 2, 0)
        grid.addLayout(minutes_row, 2, 1)

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
        search_row.addStretch(1)
        search_row.addWidget(self._search)

        self._found_told = _dim("")
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
        box.addWidget(self._how_read)
        box.addLayout(years_row)
        box.addLayout(lists_row, stretch=1)
        box.addLayout(grid)
        box.addLayout(search_row)
        box.addWidget(self._found_told)
        box.addLayout(pick_row)
        box.addWidget(self._table, stretch=2)
        box.addLayout(send_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrollable(panel))

    # ------------------------------------------------------------------
    # la libreria
    # ------------------------------------------------------------------
    def set_library(self, lib: Library) -> None:
        self._lib = lib
        frame = lib.frame
        self._vocabulary = Vocabulary.of(frame)
        self._genres.set_options(self._vocabulary.genres, keep=True)
        self._moods.set_options(self._vocabulary.moods, keep=True)
        low, high = span(frame, "bpm", describe.BPM_FLOOR, describe.BPM_CEILING)
        self._bpm.set_span(low, high)
        dated = int(frame["year"].notna().sum()) if "year" in frame else 0
        self._years_hint.setText(
            f"{dated:,} of {len(frame):,} tracks carry a year"
            + ("" if dated == len(frame) else
               " — the others have none in their tags or their name"))
        self._years_hint.setToolTip(theme.hint(
            "A filter on years keeps only tracks that carry one: a track "
            "without a year is not an 80s track, it is an unknown one. "
            "Years are read when a track goes on the map, from its tags or "
            "from a year in brackets in the file or folder name. A map "
            "built before years were read is completed from the terminal: "
            "<code>map_cli.py --years</code>."))

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
        self._how_read.setText("Reading…")
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
    # il modulo
    # ------------------------------------------------------------------
    def _show(self, query: Query, by: str) -> None:
        """Il modulo nei campi, e la riga di lettura sopra."""
        self._how_read.setText(
            f"<b>Read by {by}:</b> {query.how_read or describe.summary(query)}")
        self._years_on.setChecked(query.years is not None)
        if query.years:
            self._year_from.setValue(query.years[0])
            self._year_to.setValue(query.years[1])
        self._genres.set_checked(query.genres)
        self._moods.set_checked(query.moods)
        self._bpm_on.setChecked(query.bpm is not None)
        if query.bpm:
            self._bpm.set_values(*query.bpm)
        else:
            self._bpm.reset()
        self._title_words.setText(", ".join(query.title_words))
        self._minutes_on.setChecked(query.min_minutes is not None)
        if query.min_minutes:
            self._minutes.setValue(query.min_minutes)

    def query(self) -> Query:
        """Il modulo com'è nei campi adesso: è questo che si cerca."""
        years = ((self._year_from.value(), self._year_to.value())
                 if self._years_on.isChecked() else None)
        if years and years[0] > years[1]:
            years = (years[1], years[0])
        words = [w.strip() for w in self._title_words.text().split(",")]
        query = Query(
            years=years, genres=self._genres.checked(),
            moods=self._moods.checked(),
            bpm=self._bpm.values() if self._bpm_on.isChecked() else None,
            title_words=[w for w in words if w],
            min_minutes=(self._minutes.value() if self._minutes_on.isChecked()
                         else None))
        query.how_read = describe.summary(query)
        return query

    # ------------------------------------------------------------------
    # la ricerca
    # ------------------------------------------------------------------
    def _on_search(self) -> None:
        if self._lib is None:
            return
        lib, frame = self._lib, self._lib.frame
        query = self.query()
        found = describe.search(
            frame, lib.store.embeddings[:len(frame)], query,
            size=self._size.value(), variety=self._variety.value(),
            song_of=lambda i: song_key(Path(frame.at[i, "path"])))
        ordered = magic_sort(lib.cost, found.tracks)
        self._found = ordered
        shown = numbered_rows(frame, ordered, lib.common)
        self._table.set_tracks(
            shown, genre_colors(frame, shown["genres"], dark=theme.DARK))
        self._table.set_all_picked(True)
        self._table.setVisible(bool(ordered))
        self._add.setEnabled(bool(ordered))
        self._shelve.setEnabled(bool(ordered))
        self._found_told.setText(self._tell_found(found, len(frame)))

    @staticmethod
    def _tell_found(found: describe.Match, total: int) -> str:
        if not found.tracks:
            told = "No track matches"
            if found.no_year and found.no_year == total:
                told += " — no track on the map carries a year yet"
            return told + "."
        pieces = [f"{len(found.tracks)} track(s)",
                  f"{len(found.pool):,} of {total:,} pass the filters"]
        if found.seeds:
            pieces.append(f"{len(found.seeds)} carry the labels")
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
        name = playlist_name(self._phrase.text() or "Describe")
        if name in self._shelf.names():
            answer = QMessageBox.question(
                self, "Save the playlist", f"Overwrite «{name}» on the shelf?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.shelve_playlist.emit(name, wanted)
