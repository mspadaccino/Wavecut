"""La chiave API dell'utente: dove sta, come si legge, come si dimentica.

È dell'utente, non dell'app: la crea lui sul sito di Anthropic, la incolla
una volta, e da lì in poi non gliela si chiede più. Va nel portachiavi
del sistema — Keychain su macOS, Credential Manager su Windows — che è
fatto apposta per i segreti, e non in un file di testo accanto alle
playlist.

Dove un portachiavi non c'è (un Linux senza servizio segreti, una macchina
di prova), si ripiega su un file nella cache dell'app leggibile solo
dall'utente: peggio del portachiavi, meglio di una funzione che non parte.
Chi chiama non deve saperlo: `read`, `write`, `forget`, e `where` per
dirlo all'utente.

**La chiave non sta MAI nel bundle dell'app**: si estrae in dieci minuti,
e da lì in poi ogni lettura la paga chi ha distribuito l'app.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.analysis.map_store import default_store_dir

SERVICE = "DjCaddy"
ACCOUNT = "anthropic_api_key"

# L'ambiente vince su tutto: è come si prova dal terminale senza toccare
# il portachiavi, ed è dove uno sviluppatore la tiene comunque.
ENV_VAR = "ANTHROPIC_API_KEY"


def _fallback_file() -> Path:
    return default_store_dir().parent / "api_key"


def _keyring():
    """Il modulo keyring con un backend vero, o None."""
    try:
        import keyring
        from keyring.backends.fail import Keyring as Missing
        if isinstance(keyring.get_keyring(), Missing):
            return None
        return keyring
    except Exception:                                   # noqa: BLE001
        return None


def read() -> str | None:
    """La chiave, da dove sta; None se non c'è."""
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env
    ring = _keyring()
    if ring is not None:
        try:
            found = ring.get_password(SERVICE, ACCOUNT)
            if found:
                return found.strip() or None
        except Exception:                               # noqa: BLE001
            pass
    try:
        text = _fallback_file().read_text("utf-8").strip()
        return text or None
    except OSError:
        return None


def write(key: str) -> str:
    """Salva la chiave e dice dove l'ha messa («the system keychain» o il
    percorso del file). Una chiave vuota è un `forget`."""
    key = key.strip()
    if not key:
        forget()
        return "nowhere"
    ring = _keyring()
    if ring is not None:
        try:
            ring.set_password(SERVICE, ACCOUNT, key)
            _remove_fallback()
            return "the system keychain"
        except Exception:                               # noqa: BLE001
            pass
    path = _fallback_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, "utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return str(path)


def forget() -> None:
    ring = _keyring()
    if ring is not None:
        try:
            ring.delete_password(SERVICE, ACCOUNT)
        except Exception:                               # noqa: BLE001
            pass
    _remove_fallback()


def _remove_fallback() -> None:
    try:
        _fallback_file().unlink()
    except OSError:
        pass


def where() -> str:
    """Dove la chiave viene tenuta su questa macchina, per dirlo."""
    if os.environ.get(ENV_VAR, "").strip():
        return f"the {ENV_VAR} environment variable"
    return "the system keychain" if _keyring() is not None \
        else f"a private file, {_fallback_file()}"
