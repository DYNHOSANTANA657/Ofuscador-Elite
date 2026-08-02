from __future__ import annotations

import array
import math
import subprocess
from pathlib import Path

import pytest

from app.azure import generate_diagnostic_wav
from app.config import find_binary
from app.media import (
    MediaError,
    ffmpeg_command,
    mux_clean_video_command,
    mux_clean_video_with_voice_command,
    probe_audio,
    probe_video,
    strip_subtitle_tracks_command,
)


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stderr


def make_video(path: Path, duration: float = 3.0, with_audio: bool = True) -> None:
    command = [find_binary("ffmpeg"), "-hide_banner", "-y", "-f", "lavfi", "-i", f"color=c=navy:s=320x180:r=25:d={duration}"]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=24000:duration={duration}"]
    command += ["-c:v", "mpeg4", "-q:v", "5"]
    if with_audio:
        command += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    command.append(str(path))
    run(command)


def correlation(left: list[float], right: list[float]) -> float:
    mean_l = sum(left) / len(left)
    mean_r = sum(right) / len(right)
    numerator = sum((l - mean_l) * (r - mean_r) for l, r in zip(left, right))
    denominator = math.sqrt(sum((l - mean_l) ** 2 for l in left) * sum((r - mean_r) ** 2 for r in right))
    return numerator / denominator


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def test_processing_produces_antiphase_original_and_in_phase_voice(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    speech = tmp_path / "speech.wav"
    output = tmp_path / "output.mp4"
    make_video(source)
    generate_diagnostic_wav(speech, "Female", duration=0.7)
    source_info = probe_video(source)
    command = ffmpeg_command(source, speech, output, int(source_info["audioTracks"][0]["index"]), float(source_info["duration"]), 5)
    run(command)

    info = probe_video(output)
    assert info["videoCodec"] == source_info["videoCodec"]
    assert len(info["audioTracks"]) == 1
    assert info["audioTracks"][0]["codec"] == "aac"
    assert info["audioTracks"][0]["channels"] == 2
    assert abs(float(info["duration"]) - float(source_info["duration"])) < 0.06

    decoded = subprocess.run(
        [find_binary("ffmpeg"), "-v", "error", "-i", str(output), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "pipe:1"],
        capture_output=True,
        timeout=90,
        check=True,
    ).stdout
    samples = array.array("f")
    samples.frombytes(decoded)
    left = samples[0::2].tolist()
    right = samples[1::2].tolist()
    assert correlation(left, right) < -0.75

    original_component = [(l - r) / 2 for l, r in zip(left, right)]
    voice_component = [(l + r) / 2 for l, r in zip(left, right)]
    assert rms(original_component) > rms(voice_component) * 3
    sample_rate = 24_000
    final_voice = voice_component[-sample_rate // 2 :]
    assert rms(final_voice) > 0.001


def test_accepts_video_without_audio_for_subtitle_only_mode(tmp_path: Path) -> None:
    source = tmp_path / "silent.mp4"
    make_video(source, duration=0.5, with_audio=False)
    info = probe_video(source)
    assert info["audioTracks"] == []
    assert info["width"] == 320
    assert info["height"] == 180


def test_detects_multiple_audio_tracks(tmp_path: Path) -> None:
    source = tmp_path / "multiple.mp4"
    run([
        find_binary("ffmpeg"), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=0.6",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=0.6",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=0.6",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(source),
    ])
    info = probe_video(source)
    assert len(info["audioTracks"]) == 2
    assert info["audioTracks"][0]["index"] != info["audioTracks"][1]["index"]


def test_audio_is_padded_to_video_and_long_speech_is_trimmed(tmp_path: Path) -> None:
    source = tmp_path / "short-audio.mp4"
    speech = tmp_path / "long-speech.wav"
    output = tmp_path / "bounded.mp4"
    run([
        find_binary("ffmpeg"), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=2",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=0.5",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4", "-c:a", "aac", str(source),
    ])
    generate_diagnostic_wav(speech, "Male", duration=4.0)
    info = probe_video(source)
    run(ffmpeg_command(source, speech, output, int(info["audioTracks"][0]["index"]), float(info["duration"]), 5))
    result = probe_video(output)
    assert abs(float(result["duration"]) - 2.0) < 0.06


def test_separate_subtitle_track_is_detected_and_removed_without_reencoding(tmp_path: Path) -> None:
    base = tmp_path / "base.mp4"
    subtitle = tmp_path / "subtitle.srt"
    source = tmp_path / "with-subtitle.mp4"
    output = tmp_path / "without-subtitle.mp4"
    make_video(base, duration=1.2)
    subtitle.write_text("1\n00:00:00,100 --> 00:00:01,000\nLegenda de teste\n", encoding="utf-8")
    run([find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(base), "-i", str(subtitle), "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", str(source)])
    before = probe_video(source)
    assert len(before["subtitleTracks"]) == 1
    run(strip_subtitle_tracks_command(source, output, float(before["duration"])))
    after = probe_video(output)
    assert after["subtitleTracks"] == []
    assert after["videoCodec"] == before["videoCodec"]
    assert after["audioTracks"][0]["codec"] == before["audioTracks"][0]["codec"]


def make_longer_audio_video(path: Path) -> None:
    """Vídeo de 3s com áudio de 3,4s — o contêiner passa a valer mais que o vídeo."""
    run([
        find_binary("ffmpeg"), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30000/1001:d=3",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=3.4",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-bf", "3", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ])


def test_probe_separates_container_duration_from_video_duration(tmp_path: Path) -> None:
    source = tmp_path / "audio-mais-longo.mp4"
    make_longer_audio_video(source)
    info = probe_video(source)
    # É esta diferença que quebrava a verificação final: a antiga comparava a duração do
    # vídeo reconstruído contra a duração do contêiner, que aqui é a da faixa de áudio.
    assert float(info["duration"]) > float(info["videoDuration"]) + 0.2
    assert int(info["frameCount"]) > 0
    assert "/" in str(info["frameRateRational"])
    assert abs(float(info["videoDuration"]) - int(info["frameCount"]) / float(info["fps"])) < 0.05


def test_mux_clean_video_does_not_truncate_the_reconstructed_video(tmp_path: Path) -> None:
    source = tmp_path / "origem.mp4"
    clean = tmp_path / "limpo.mp4"
    output = tmp_path / "final.mp4"
    make_longer_audio_video(source)
    info = probe_video(source)
    run([find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(source), "-map", "0:v:0", "-an", "-c:v", "copy", str(clean)])
    # Passar a duração do contêiner aqui era o que encurtava o resultado: com `-c:v copy`
    # o `-t` só corta. O comando precisa ignorar esse valor.
    run(mux_clean_video_command(clean, source, output, float(info["duration"])))
    final = probe_video(output)
    assert int(final["frameCount"]) == int(info["frameCount"])
    assert abs(float(final["videoDuration"]) - float(info["videoDuration"])) < 0.05
    assert "-t" not in mux_clean_video_command(clean, source, output, float(info["duration"]))


def test_probe_audio_accepts_mp3_and_rejects_files_without_audio(tmp_path: Path) -> None:
    music = tmp_path / "trilha.mp3"
    run([find_binary("ffmpeg"), "-hide_banner", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2.5", "-c:a", "libmp3lame", str(music)])
    info = probe_audio(music)
    assert info["codec"] == "mp3"
    assert abs(float(info["duration"]) - 2.5) < 0.15

    silent = tmp_path / "sem-audio.mp4"
    make_video(silent, duration=0.5, with_audio=False)
    with pytest.raises(MediaError, match="faixa de áudio"):
        probe_audio(silent)


def test_user_supplied_audio_is_mixed_like_the_reference_ffmpeg_line(tmp_path: Path) -> None:
    """Reproduz o comando do arquivo de referência: áudio do vídeo a 1.0 e uma
    trilha própria a 5%, com a antifase do aplicativo por cima."""
    source = tmp_path / "source.mp4"
    music = tmp_path / "trilha.mp3"
    output = tmp_path / "final.mp4"
    make_video(source, duration=3.0)
    run([find_binary("ffmpeg"), "-hide_banner", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1.0", "-c:a", "libmp3lame", str(music)])
    info = probe_video(source)
    # O MP3 entra no lugar do WAV da voz: o mesmo caminho serve para os dois.
    run(ffmpeg_command(source, music, output, int(info["audioTracks"][0]["index"]), float(info["duration"]), 5))
    result = probe_video(output)
    assert result["audioTracks"][0]["channels"] == 2
    assert abs(float(result["duration"]) - float(info["duration"])) < 0.06

    decoded = subprocess.run(
        [find_binary("ffmpeg"), "-v", "error", "-i", str(output), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "pipe:1"],
        capture_output=True, timeout=90, check=True,
    ).stdout
    samples = array.array("f")
    samples.frombytes(decoded)
    left, right = samples[0::2].tolist(), samples[1::2].tolist()
    assert correlation(left, right) < -0.75
    # A trilha de 1s foi repetida pelo `-stream_loop -1` até o fim dos 3s de vídeo.
    tail = [(l + r) / 2 for l, r in zip(left[-24_000:], right[-24_000:])]
    assert rms(tail) > 0.001


def test_combined_clean_video_keeps_antiphase_mix(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    clean = tmp_path / "clean.mp4"
    speech = tmp_path / "speech.wav"
    output = tmp_path / "combined.mp4"
    make_video(source, duration=1.2)
    run([find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(source), "-map", "0:v:0", "-an", "-c:v", "copy", str(clean)])
    generate_diagnostic_wav(speech, "Female", duration=0.4)
    before = probe_video(source)
    run(mux_clean_video_with_voice_command(clean, source, speech, output, int(before["audioTracks"][0]["index"]), float(before["duration"]), 5))
    after = probe_video(output)
    assert after["videoCodec"] == before["videoCodec"]
    assert len(after["audioTracks"]) == 1
    assert after["audioTracks"][0]["channels"] == 2
    decoded = subprocess.run([find_binary("ffmpeg"), "-v", "error", "-i", str(output), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "pipe:1"], capture_output=True, timeout=90, check=True).stdout
    samples = array.array("f")
    samples.frombytes(decoded)
    assert correlation(samples[0::2].tolist(), samples[1::2].tolist()) < -0.7
