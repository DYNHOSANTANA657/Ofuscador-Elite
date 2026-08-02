from __future__ import annotations

import keyring

SERVICE = "OfuscadorElite.AzureSpeech"


def get_azure_credentials() -> tuple[str | None, str | None]:
    return keyring.get_password(SERVICE, "subscription-key"), keyring.get_password(SERVICE, "region")


def set_azure_credentials(key: str, region: str) -> None:
    keyring.set_password(SERVICE, "subscription-key", key)
    keyring.set_password(SERVICE, "region", region)


def delete_azure_credentials() -> None:
    for username in ("subscription-key", "region"):
        try:
            keyring.delete_password(SERVICE, username)
        except keyring.errors.PasswordDeleteError:
            pass
