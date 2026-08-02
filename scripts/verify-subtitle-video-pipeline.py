from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa um vídeo curto pela remoção gravada completa.")
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="ofuscador-video-verify-") as temporary:
        root = Path(temporary).resolve()
        os.environ["OFUSCADOR_DATA_DIR"] = str(root / "data")
        os.environ["OFUSCADOR_TEMP_DIR"] = str(root / "temp")
        from app.config import find_binary
        from app.media import mux_clean_video_command, probe_video
        from app.subtitle_models import SubtitleModelManager
        from app.subtitle_processing import remove_burned_subtitles

        manager = SubtitleModelManager()
        manager.install_import(package)
        # 20s a 30000/1001 com B-frames e GOP longo, e áudio propositalmente mais longo
        # que o vídeo. O diagnóstico anterior usava 8 quadros a 8 fps: nessa escala
        # nenhuma perda de quadro nem a diferença entre duração de contêiner e de vídeo
        # chega a aparecer, e a verificação passava com a pipeline quebrada.
        source = root / "source.mp4"
        legend = "drawtext=text='LEGENDA DE TESTE':fontcolor=white:fontsize=22:x=(w-tw)/2:y=h-52"
        result = subprocess.run([
            find_binary("ffmpeg"), "-hide_banner", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30000/1001:d=20",
            "-f", "lavfi", "-i", "sine=frequency=260:duration=20.4",
            "-map", "0:v:0", "-map", "1:a:0", "-vf", legend,
            "-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-g", "250", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(source),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        if result.returncode:
            raise RuntimeError(result.stderr)
        info = probe_video(source)
        clean = root / "clean.mp4"
        region = {"id": "manual", "x": 0.18, "y": 0.80, "width": 0.64, "height": 0.16, "startMs": 0, "endMs": 20_400, "source": "manual", "enabled": True}
        stats = remove_burned_subtitles(source, clean, info, [region], manager)
        output = root / "output.mp4"
        mux = subprocess.run(mux_clean_video_command(clean, source, output, float(info["duration"])), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        if mux.returncode:
            raise RuntimeError(mux.stdout + mux.stderr)
        final = probe_video(output)
        tolerance = max(2 / float(info["fps"]), 0.15)
        checks = {
            "playable": output.is_file() and output.stat().st_size > 0,
            "resolution": final["width"] == info["width"] and final["height"] == info["height"],
            "audioPreserved": len(final["audioTracks"]) == len(info["audioTracks"]),
            "subtitlesAbsent": final["subtitleTracks"] == [],
            "allFramesRead": int(stats["frames"]) == int(info["frameCount"]),
            "frameCountPreserved": int(final["frameCount"]) == int(info["frameCount"]),
            "videoDurationWithinTolerance": abs(float(final["videoDuration"]) - float(info["videoDuration"])) <= tolerance,
            "codec": final["videoCodec"] == "h264",
        }
        detail = {
            "sourceFrames": int(info["frameCount"]), "readFrames": int(stats["frames"]), "finalFrames": int(final["frameCount"]),
            "sourceVideoDuration": float(info["videoDuration"]), "finalVideoDuration": float(final["videoDuration"]),
            "sourceContainerDuration": float(info["duration"]), "finalContainerDuration": float(final["duration"]),
        }
        if not all(checks.values()):
            raise RuntimeError(json.dumps({"checks": checks, "detail": detail}))
        print(json.dumps({"checks": checks, "detail": detail}))


if __name__ == "__main__":
    main()
