from __future__ import annotations

import html
import http.client
import json
import math
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

DIAGNOSTIC_VOICES = [
    {"shortName": "diagnostico-feminino", "displayName": "Voz de teste feminina", "gender": "Female", "locale": "pt-BR", "diagnostic": True},
    {"shortName": "diagnostico-masculino", "displayName": "Voz de teste masculina", "gender": "Male", "locale": "pt-BR", "diagnostic": True},
]


class AzureSpeechError(RuntimeError):
    pass


def _retry_delay(error: urllib.error.HTTPError | None, attempt: int) -> float:
    if error is not None:
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                return min(5.0, max(0.5, float(retry_after)))
            except ValueError:
                pass
    return 0.6 * (attempt + 1)


def _request(url: str, key: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 30) -> bytes:
    request_headers = {"Ocp-Apim-Subscription-Key": key, "User-Agent": "OfuscadorElite/1.0"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST" if data is not None else "GET")
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", "replace")[:500]
            if exc.code in {401, 403}:
                raise AzureSpeechError("A chave ou a região Azure não foi aceita.") from exc
            if exc.code in {408, 429, 500, 502, 503, 504} and attempt == 0:
                time.sleep(_retry_delay(exc, attempt))
                continue
            if exc.code == 429:
                raise AzureSpeechError("O limite temporário da conta Azure foi atingido. Tente novamente em instantes.") from exc
            raise AzureSpeechError(f"A Azure respondeu com erro {exc.code}: {details or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            if attempt == 0:
                time.sleep(_retry_delay(None, attempt))
                continue
            raise AzureSpeechError("Não foi possível conectar ao serviço Azure Speech após uma nova tentativa.") from exc
    raise AzureSpeechError("Não foi possível conectar ao serviço Azure Speech.")


def list_voices(key: str, region: str) -> list[dict[str, object]]:
    url = f"https://{region.strip()}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    raw = json.loads(_request(url, key).decode("utf-8"))
    voices = []
    for item in raw:
        locale = str(item.get("Locale", ""))
        gender = str(item.get("Gender", ""))
        if locale.lower() != "pt-br" or gender not in {"Male", "Female"}:
            continue
        voices.append({
            "shortName": str(item.get("ShortName", "")),
            "displayName": str(item.get("LocalName") or item.get("DisplayName") or item.get("ShortName", "")),
            "gender": gender,
            "locale": locale,
            "diagnostic": False,
        })
    return sorted(voices, key=lambda voice: (str(voice["gender"]), str(voice["displayName"])))


def build_ssml(text: str, voice: str) -> str:
    safe_text = html.escape(text.strip(), quote=False)
    safe_voice = html.escape(voice.strip(), quote=True)
    return f'<speak version="1.0" xml:lang="pt-BR"><voice name="{safe_voice}">{safe_text}</voice></speak>'


def synthesize_azure(text: str, voice: str, key: str, region: str, destination: Path) -> None:
    url = f"https://{region.strip()}.tts.speech.microsoft.com/cognitiveservices/v1"
    audio = _request(
        url,
        key,
        data=build_ssml(text, voice).encode("utf-8"),
        headers={
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
        },
        timeout=90,
    )
    if len(audio) < 44 or audio[:4] != b"RIFF":
        raise AzureSpeechError("A Azure não devolveu um áudio WAV válido.")
    destination.write_bytes(audio)


def generate_diagnostic_wav(destination: Path, gender: str = "Female", duration: float = 2.0) -> None:
    sample_rate = 24_000
    frequency = 660.0 if gender == "Female" else 440.0
    total = int(sample_rate * duration)
    fade = int(sample_rate * 0.08)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total):
            envelope = min(1.0, index / max(1, fade), (total - index) / max(1, fade))
            pulse = 0.55 + 0.45 * math.sin(2 * math.pi * 2.0 * index / sample_rate)
            value = int(32767 * 0.28 * envelope * pulse * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        output.writeframes(frames)


class VoiceCache:
    def __init__(self) -> None:
        self._key = ""
        self._expires = 0.0
        self._voices: list[dict[str, object]] = []

    def get(self, key: str, region: str) -> list[dict[str, object]]:
        cache_key = f"{region}:{key[-6:]}"
        if cache_key == self._key and time.time() < self._expires:
            return self._voices
        voices = list_voices(key, region)
        self._key = cache_key
        self._expires = time.time() + 3600
        self._voices = voices
        return voices

    def clear(self) -> None:
        self._key = ""
        self._expires = 0.0
        self._voices = []
