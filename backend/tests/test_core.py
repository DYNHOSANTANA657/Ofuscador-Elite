from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.azure import build_ssml, generate_diagnostic_wav
from app.main import app
from app.media import ffmpeg_command, safe_stem
from app.security import SessionSecurity, is_loopback


def test_ssml_escapes_user_text_and_voice() -> None:
    ssml = build_ssml('Olá <teste> & tudo "bem"', 'pt-BR-FranciscaNeural" on=x')
    assert "Olá &lt;teste&gt; &amp; tudo \"bem\"" in ssml
    assert 'name="pt-BR-FranciscaNeural&quot; on=x"' in ssml


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("meu vídeo.mp4", "meu vídeo"),
        ("../segredo?.mp4", "segredo"),
        ("CON:<teste>|.mp4", "CON-teste"),
        ("   .mp4", "video"),
    ],
)
def test_safe_stem(filename: str, expected: str) -> None:
    assert safe_stem(filename) == expected


def test_ffmpeg_command_is_structured_and_bounded(tmp_path: Path) -> None:
    command = ffmpeg_command(tmp_path / "vídeo.mp4", tmp_path / "voz.wav", tmp_path / "saída.mp4", 2, 9.75, 25)
    assert isinstance(command, list)
    assert "shell" not in command
    graph = command[command.index("-filter_complex") + 1]
    assert "normalize=0" in graph
    assert "alimiter=limit=0.98" in graph
    assert "volume=0.2500" in graph
    assert "c1=-1*c0" in graph
    assert "apad=whole_dur=9.750000" in graph
    assert "-c:v" in command and command[command.index("-c:v") + 1] == "copy"


def test_session_tokens_and_loopback() -> None:
    security = SessionSecurity(network_mode=True, pin="123456")
    token = security.issue()
    assert security.verify(token)
    assert not security.verify(token[:-2] + "xx")
    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert not is_loopback("192.168.1.10")


def test_pin_attempts_are_rate_limited() -> None:
    security = SessionSecurity(network_mode=True, pin="123456")
    for second in range(5):
        allowed, retry_after = security.authenticate("192.168.1.20", "000000", now=float(second))
        assert not allowed
        assert retry_after == 0
    allowed, retry_after = security.authenticate("192.168.1.20", "123456", now=5.0)
    assert not allowed
    assert retry_after > 0
    allowed, retry_after = security.authenticate("192.168.1.20", "123456", now=65.0)
    assert allowed
    assert retry_after == 0


def test_api_validation_and_voice_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.main.credentials", lambda: (None, None))
    monkeypatch.setattr("app.main.piper_available", lambda: True)
    client = TestClient(app)
    status = client.get("/api/config/status")
    assert status.status_code == 200
    assert status.json()["piperAvailable"] is True
    assert status.json()["azureConfigured"] is False
    assert status.json()["defaultProvider"] == "piper"
    voices = client.get("/api/voices?provider=piper&gender=Male")
    assert voices.status_code == 200
    assert voices.json()["voices"][0]["provider"] == "piper"
    assert client.get("/api/voices?provider=piper&gender=Female").json()["voices"] == []
    assert client.get("/api/voices?provider=azure&gender=Female").json()["voices"] == []
    invalid = client.post("/api/jobs", json={"uploadId": "a" * 32, "text": "x" * 5001, "voice": "v", "gender": "Male", "provider": "piper", "volumePercent": 5, "audioStreamIndex": 1})
    assert invalid.status_code == 422
    no_removal = client.post("/api/jobs", json={"uploadId": "a" * 32, "mode": "subtitles", "subtitleRemoval": {}})
    assert no_removal.status_code == 422
    status_models = client.get("/api/subtitle-model/status")
    assert status_models.status_code == 200
    assert "installed" in status_models.json()


def test_piper_preview_returns_browser_playable_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_piper(_text: str, _voice: str, destination: Path) -> None:
        generate_diagnostic_wav(destination, "Male", duration=0.25)

    monkeypatch.setattr("app.main.synthesize_piper", fake_piper)
    client = TestClient(app)
    response = client.post("/api/preview", json={"text": "Teste", "voice": "pt_BR-faber-medium", "gender": "Male", "provider": "piper"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content[:4] == b"RIFF"
    assert len(response.content) > 48


def test_remote_client_requires_session() -> None:
    client = TestClient(app, client=("192.168.50.20", 5000))
    assert client.get("/api/config/status").status_code == 401


def test_cannot_delete_upload_while_job_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    upload_id = "b" * 32
    monkeypatch.setattr("app.main.UPLOADS.get", lambda candidate: object() if candidate == upload_id else None)
    monkeypatch.setattr("app.main.JOBS.upload_in_use", lambda candidate: candidate == upload_id)
    client = TestClient(app)
    response = client.delete(f"/api/uploads/{upload_id}")
    assert response.status_code == 409
    assert "processado" in response.json()["detail"]
