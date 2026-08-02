import io
import json
import urllib.error

import pytest

from app import azure


def test_voice_list_filters_pt_br_and_gender(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {"ShortName": "pt-BR-FranciscaNeural", "LocalName": "Francisca", "Locale": "pt-BR", "Gender": "Female"},
        {"ShortName": "pt-BR-AntonioNeural", "LocalName": "Antônio", "Locale": "pt-BR", "Gender": "Male"},
        {"ShortName": "en-US-JennyNeural", "LocalName": "Jenny", "Locale": "en-US", "Gender": "Female"},
    ]
    monkeypatch.setattr(azure, "_request", lambda *args, **kwargs: json.dumps(payload).encode())
    voices = azure.list_voices("key", "brazilsouth")
    assert {voice["shortName"] for voice in voices} == {"pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"}


def test_invalid_azure_key_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("https://example.test", 401, "Unauthorized", {}, io.BytesIO(b"invalid"))

    monkeypatch.setattr(azure.urllib.request, "urlopen", reject)
    with pytest.raises(azure.AzureSpeechError, match="chave ou a região"):
        azure.list_voices("bad", "wrong")
    assert calls == 1


def test_temporary_connection_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"recovered"

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary")
        return Response()

    monkeypatch.setattr(azure.urllib.request, "urlopen", request)
    monkeypatch.setattr(azure.time, "sleep", lambda _seconds: None)

    assert azure._request("https://example.test", "key") == b"recovered"
    assert calls == 2


def test_temporary_http_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"recovered"

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("https://example.test", 503, "Unavailable", {}, io.BytesIO(b"temporary"))
        return Response()

    monkeypatch.setattr(azure.urllib.request, "urlopen", request)
    monkeypatch.setattr(azure.time, "sleep", lambda _seconds: None)

    assert azure._request("https://example.test", "key") == b"recovered"
    assert calls == 2
