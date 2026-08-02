from pathlib import Path

import pytest

from app import piper_local


def test_piper_voice_is_local_pt_br_and_male() -> None:
    voice = piper_local.PIPER_VOICES[0]
    assert voice["provider"] == "piper"
    assert voice["locale"] == "pt-BR"
    assert voice["gender"] == "Male"
    assert voice["local"] is True


def test_unknown_piper_voice_is_rejected() -> None:
    with pytest.raises(piper_local.PiperSpeechError, match="não existe"):
        piper_local.voice_model_path("../../modelo")


def test_piper_wav_validation_with_fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    model = voices / "pt_BR-faber-medium.onnx"
    model.write_bytes(b"model")
    model.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(piper_local, "resource_root", lambda: tmp_path)

    class FakeVoice:
        @classmethod
        def load(cls, *_args, **_kwargs):
            return cls()

        def synthesize_wav(self, _text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(b"\x00\x00" * 100)

    import piper

    monkeypatch.setattr(piper, "PiperVoice", FakeVoice)
    piper_local._VOICE_CACHE.clear()
    output = tmp_path / "speech.wav"
    piper_local.synthesize_piper("Olá", "pt_BR-faber-medium", output)
    assert output.read_bytes()[:4] == b"RIFF"
