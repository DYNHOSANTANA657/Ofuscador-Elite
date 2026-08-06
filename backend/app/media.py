from __future__ import annotations

import json
import math
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from .config import find_binary


class MediaError(RuntimeError):
    pass


def _rate(value: object) -> float:
    try:
        text = str(value or "0/0")
        if text in {"0/0", "N/A"}:
            return 0.0
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rational_text(value: object, fallback: float) -> str:
    """Taxa de quadros como fração exata (ex.: 30000/1001) para o FFmpeg.

    Passar 29.97002997 faz o FFmpeg reaproximar a fração e acumular desvio ao longo
    de milhares de quadros; a fração original do contêiner não tem esse problema.
    """
    text = str(value or "").strip()
    if "/" in text and _rate(text) > 0:
        return text
    fraction = Fraction(fallback).limit_denominator(1_000_000)
    return f"{fraction.numerator}/{fraction.denominator}"


def _stream_float(stream: dict[str, object], key: str) -> float:
    try:
        value = float(str(stream.get(key, "")))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def probe_video(path: Path) -> dict[str, object]:
    command = [find_binary("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError("O vídeo não pôde ser analisado pelo FFprobe.") from exc
    if result.returncode != 0:
        raise MediaError("O arquivo não é um MP4 válido ou está corrompido.")
    try:
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError("Não foi possível identificar a duração do vídeo.") from exc
    streams = list(data.get("streams", []))
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    subtitles = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    if not videos or not math.isfinite(duration) or duration <= 0:
        raise MediaError("O arquivo precisa conter uma faixa de vídeo válida.")

    video = videos[0]
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    avg_fps = _rate(video.get("avg_frame_rate"))
    nominal_fps = _rate(video.get("r_frame_rate"))
    fps = avg_fps or nominal_fps
    if width <= 0 or height <= 0 or fps <= 0:
        raise MediaError("Não foi possível identificar a resolução ou o FPS do vídeo.")
    variable_frame_rate = bool(avg_fps and nominal_fps and abs(avg_fps - nominal_fps) / max(avg_fps, nominal_fps) > 0.001)
    frame_rate_rational = _rational_text(video.get("avg_frame_rate") if avg_fps else video.get("r_frame_rate"), fps)
    video_duration = _stream_float(video, "duration")
    if not video_duration:
        # MKV e alguns MP4 não trazem duration por stream; o contêiner é a melhor estimativa.
        video_duration = duration
    try:
        frame_count = int(str(video.get("nb_frames", "") or 0))
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0:
        frame_count = int(round(video_duration * fps))
    transfer = str(video.get("color_transfer", "") or "").lower()
    primaries = str(video.get("color_primaries", "") or "").lower()
    hdr = transfer in {"smpte2084", "arib-std-b67"} or primaries == "bt2020"

    audio_tracks: list[dict[str, object]] = []
    for order, stream in enumerate(audios, 1):
        tags = stream.get("tags", {}) or {}
        audio_tracks.append({
            "index": int(stream["index"]),
            "order": order,
            "codec": str(stream.get("codec_name", "desconhecido")),
            "channels": int(stream.get("channels", 0) or 0),
            "language": str(tags.get("language", "und")),
            "title": str(tags.get("title", "")),
        })
    subtitle_tracks: list[dict[str, object]] = []
    for order, stream in enumerate(subtitles, 1):
        tags = stream.get("tags", {}) or {}
        disposition = stream.get("disposition", {}) or {}
        subtitle_tracks.append({
            "index": int(stream["index"]),
            "order": order,
            "codec": str(stream.get("codec_name", "desconhecido")),
            "language": str(tags.get("language", "und")),
            "title": str(tags.get("title", "")),
            "default": bool(disposition.get("default", 0)),
            "forced": bool(disposition.get("forced", 0)),
        })
    return {
        "duration": duration,
        "videoDuration": video_duration,
        "frameCount": frame_count,
        "frameRateRational": frame_rate_rational,
        "videoStreamIndex": int(video["index"]),
        "videoCodec": str(video.get("codec_name", "")),
        "width": width,
        "height": height,
        "fps": fps,
        "averageFrameRate": avg_fps,
        "nominalFrameRate": nominal_fps,
        "variableFrameRate": variable_frame_rate,
        "pixelFormat": str(video.get("pix_fmt", "")),
        "colorSpace": str(video.get("color_space", "")),
        "colorTransfer": transfer,
        "colorPrimaries": primaries,
        "hdr": hdr,
        "audioTracks": audio_tracks,
        "subtitleTracks": subtitle_tracks,
    }


def probe_audio(path: Path) -> dict[str, object]:
    """Valida um arquivo de áudio enviado pelo usuário para ser mixado no vídeo."""
    command = [find_binary("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError("O áudio não pôde ser analisado pelo FFprobe.") from exc
    if result.returncode != 0:
        raise MediaError("O arquivo de áudio está corrompido ou em um formato não reconhecido.")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("Não foi possível analisar o arquivo de áudio.") from exc
    streams = list(data.get("streams", []))
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audios:
        raise MediaError("Este arquivo não contém nenhuma faixa de áudio.")
    audio = audios[0]
    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        duration = _stream_float(audio, "duration")
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("Não foi possível identificar a duração do áudio.")
    return {
        "duration": duration,
        "codec": str(audio.get("codec_name", "desconhecido")),
        "channels": int(audio.get("channels", 0) or 0),
        "sampleRate": int(str(audio.get("sample_rate", "0") or 0)),
    }


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", "-", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .-")
    return stem[:120] or "video"


def unique_output(directory: Path, filename: str, subtitles: bool = False, suffix: str | None = None) -> Path:
    if suffix is None:
        suffix = "-sem-legendas-processado" if subtitles else "-processado"
    base = safe_stem(filename) + suffix
    candidate = directory / f"{base}.mp4"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{base}-{counter}.mp4"
        counter += 1
    return candidate


def ffmpeg_command(video: Path, speech: Path, output: Path, audio_stream_index: int, duration: float, volume_percent: int, keep_subtitles: bool = False) -> list[str]:
    volume = max(0, min(25, volume_percent)) / 100.0
    filter_graph = (
        f"[0:{audio_stream_index}]aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f},"
        "pan=stereo|c0=c0|c1=-1*c0[original];"
        f"[1:a:0]volume={volume:.4f},pan=stereo|c0=c0|c1=c0[voice];"
        "[original][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.98:level=disabled[audio]"
    )
    command = [
        find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(video), "-stream_loop", "-1", "-i", str(speech),
        "-filter_complex", filter_graph, "-map", "0:v:0", "-map", "[audio]",
    ]
    if keep_subtitles:
        command += ["-map", "0:s?", "-c:s", "copy"]
    else:
        command += ["-sn"]
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output)]
    return command


def strip_subtitle_tracks_command(video: Path, output: Path, duration: float) -> list[str]:
    return [
        find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(video), "-map", "0:v:0", "-map", "0:a?",
        "-c", "copy", "-sn", "-t", f"{duration:.6f}", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output),
    ]


def mux_clean_video_command(clean_video: Path, original: Path, output: Path, duration: float, remove_subtitles: bool = True) -> list[str]:
    """Junta o vídeo já reconstruído com o áudio original, sem recodificar nada.

    O parâmetro `duration` existe só para registro no log: com `-c:v copy` o `-t`
    apenas corta e nunca estende, então um vídeo limpo levemente curto virava um
    arquivo final curto. O vídeo reconstruído já tem a contagem exata de quadros.
    """
    command = [
        find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(clean_video), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?",
    ]
    if remove_subtitles:
        command += ["-sn"]
    else:
        command += ["-map", "1:s?", "-c:s", "copy"]
    command += ["-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output)]
    return command


def mux_clean_video_with_voice_command(
    clean_video: Path,
    original: Path,
    speech: Path,
    output: Path,
    audio_stream_index: int,
    duration: float,
    volume_percent: int,
    remove_subtitles: bool = True,
) -> list[str]:
    volume = max(0, min(25, volume_percent)) / 100.0
    filter_graph = (
        f"[1:{audio_stream_index}]aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f},"
        "pan=stereo|c0=c0|c1=-1*c0[original];"
        f"[2:a:0]volume={volume:.4f},pan=stereo|c0=c0|c1=c0[voice];"
        "[original][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.98:level=disabled[audio]"
    )
    command = [
        find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(clean_video), "-i", str(original),
        "-stream_loop", "-1", "-i", str(speech), "-filter_complex", filter_graph,
        "-map", "0:v:0", "-map", "[audio]",
    ]
    if remove_subtitles:
        command += ["-sn"]
    else:
        command += ["-map", "1:s?", "-c:s", "copy"]
    # Sem `-t`: o `apad`/`atrim` acima já limita o áudio a `duration` e o vídeo
    # reconstruído já tem a duração certa. `-t` só encurtaria o resultado.
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output)]
    return command


def has_h264_encoder() -> bool:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run([find_binary("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30, check=False, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "libx264" in result.stdout
