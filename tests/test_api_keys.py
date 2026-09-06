"""La chiave API: ambiente, portachiavi, e il file di ripiego."""

from core.analysis import api_keys


def _no_keyring(monkeypatch, tmp_path):
    monkeypatch.setattr(api_keys, "_keyring", lambda: None)
    monkeypatch.setattr(api_keys, "_fallback_file", lambda: tmp_path / "api_key")
    monkeypatch.delenv(api_keys.ENV_VAR, raising=False)


def test_without_a_keyring_the_key_lives_in_a_private_file(monkeypatch, tmp_path):
    _no_keyring(monkeypatch, tmp_path)
    assert api_keys.read() is None
    assert api_keys.write("  sk-ant-test  ") == str(tmp_path / "api_key")
    assert api_keys.read() == "sk-ant-test"
    assert (tmp_path / "api_key").stat().st_mode & 0o077 == 0
    assert "private file" in api_keys.where()
    api_keys.forget()
    assert api_keys.read() is None


def test_an_empty_key_is_a_forget(monkeypatch, tmp_path):
    _no_keyring(monkeypatch, tmp_path)
    api_keys.write("sk-ant-test")
    assert api_keys.write("   ") == "nowhere"
    assert api_keys.read() is None


def test_the_environment_wins(monkeypatch, tmp_path):
    _no_keyring(monkeypatch, tmp_path)
    api_keys.write("sk-file")
    monkeypatch.setenv(api_keys.ENV_VAR, "sk-env")
    assert api_keys.read() == "sk-env"
    assert api_keys.ENV_VAR in api_keys.where()


class _Ring:
    def __init__(self) -> None:
        self.saved: dict = {}

    def get_password(self, service, account):
        return self.saved.get((service, account))

    def set_password(self, service, account, key):
        self.saved[(service, account)] = key

    def delete_password(self, service, account):
        self.saved.pop((service, account), None)


def test_with_a_keyring_the_key_goes_there_and_the_file_goes_away(monkeypatch, tmp_path):
    ring = _Ring()
    monkeypatch.setattr(api_keys, "_keyring", lambda: ring)
    monkeypatch.setattr(api_keys, "_fallback_file", lambda: tmp_path / "api_key")
    monkeypatch.delenv(api_keys.ENV_VAR, raising=False)
    (tmp_path / "api_key").write_text("old")
    assert api_keys.write("sk-ring") == "the system keychain"
    assert not (tmp_path / "api_key").exists()
    assert api_keys.read() == "sk-ring"
    assert api_keys.where() == "the system keychain"
    api_keys.forget()
    assert api_keys.read() is None
