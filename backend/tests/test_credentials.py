import pytest

from app import credentials


class MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str):
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self.values[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self.values:
            raise credentials.keyring.errors.PasswordDeleteError("missing")
        del self.values[key]


def test_credentials_use_keyring_without_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    vault = MemoryKeyring()
    monkeypatch.setattr(credentials, "keyring", vault)
    credentials.set_azure_credentials("secret-key", "brazilsouth")
    assert credentials.get_azure_credentials() == ("secret-key", "brazilsouth")
    credentials.delete_azure_credentials()
    assert credentials.get_azure_credentials() == (None, None)
