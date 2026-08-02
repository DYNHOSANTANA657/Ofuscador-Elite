from __future__ import annotations

import threading
import wave
from pathlib import Path

from .config import resource_root

PIPER_VOICES: list[dict[str, object]] = [
    {
        "shortName": "pt_BR-faber-medium",
        "displayName": "Faber — masculina (local)",
        "gender": "Male",
        "locale": "pt-BR",
        "provider": "piper",
        "local": True,
    }
]

_VOICE_NAMES = {str(voice["shortName"]) for voice in PIPER_VOICES}
_VOICE_CACHE: dict[str, object] = {}
_VOICE_LOCK = threading.RLock()


class PiperSpeechError(RuntimeError):
    pass


def voice_model_path(short_name: str) -> Path:
    if short_name not in _VOICE_NAMES:
        raise PiperSpeechError("A voz local escolhida não existe neste pacote.")
    return resource_root() / "voices" / f"{short_name}.onnx"


def piper_available() -> bool:
    try:
        model = voice_model_path(str(PIPER_VOICES[0]["shortName"]))
        config = model.with_suffix(model.suffix + ".json")
        if not model.is_file() or not config.is_file():
            return False
        from piper import PiperVoice  # noqa: F401

        return True
    except Exception:
        return False


def synthesize_piper(text: str, short_name: str, destination: Path) -> None:
    model = voice_model_path(short_name)
    config = model.with_suffix(model.suffix + ".json")
    if not model.is_file() or not config.is_file():
        raise PiperSpeechError("O modelo de voz local do Piper não foi encontrado no pacote.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from piper import PiperVoice

        with _VOICE_LOCK:
            voice = _VOICE_CACHE.get(short_name)
            if voice is None:
                voice = PiperVoice.load(model, config_path=config)
                _VOICE_CACHE[short_name] = voice
            with wave.open(str(destination), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
    except PiperSpeechError:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise PiperSpeechError("O Piper não conseguiu transformar o texto em fala local.") from exc

    try:
        with destination.open("rb") as audio:
            if audio.read(4) != b"RIFF" or destination.stat().st_size < 48:
                raise OSError
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise PiperSpeechError("O Piper não gerou um áudio WAV válido.") from exc
