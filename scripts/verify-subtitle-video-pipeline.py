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
        import cv2
        import numpy as np
        from app.config import find_binary
        from app.media import mux_clean_video_command, probe_video
        from app.subtitle_models import SubtitleModelManager
        from app.subtitle_processing import remove_burned_subtitles

        manager = SubtitleModelManager()
        manager.install_import(package)
        visual = root / "visual.mp4"
        writer = cv2.VideoWriter(str(visual), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (320, 180))
        if not writer.isOpened():
            raise RuntimeError("OpenCV não abriu o vídeo de diagnóstico.")
        for index in range(8):
            frame = np.full((180, 320, 3), (55 + index, 105, 145), dtype=np.uint8)
            cv2.putText(frame, "LEGENDA", (95, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        source = root / "source.mp4"
        result = subprocess.run([
            find_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(visual), "-f", "lavfi", "-i", "sine=frequency=260:duration=1",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(source),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr)
        info = probe_video(source)
        clean = root / "clean.mp4"
        region = {"id": "manual", "x": 0.22, "y": 0.70, "width": 0.56, "height": 0.25, "startMs": 0, "endMs": 1000, "source": "manual", "enabled": True}
        remove_burned_subtitles(source, clean, info, [region], manager)
        output = root / "output.mp4"
        mux = subprocess.run(mux_clean_video_command(clean, source, output, float(info["duration"])), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        if mux.returncode:
            raise RuntimeError(mux.stdout + mux.stderr)
        final = probe_video(output)
        tolerance = 1 / float(info["fps"]) + 0.04
        checks = {
            "playable": output.is_file() and output.stat().st_size > 0,
            "resolution": final["width"] == info["width"] and final["height"] == info["height"],
            "audioPreserved": len(final["audioTracks"]) == len(info["audioTracks"]),
            "subtitlesAbsent": final["subtitleTracks"] == [],
            "durationWithinFrame": abs(float(final["duration"]) - float(info["duration"])) <= tolerance,
            "codec": final["videoCodec"] == "h264",
        }
        if not all(checks.values()):
            raise RuntimeError(json.dumps(checks))
        print(json.dumps(checks))


if __name__ == "__main__":
    main()
