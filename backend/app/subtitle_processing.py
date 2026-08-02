from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from .config import find_binary
from .subtitle_models import SubtitleModelManager


class SubtitleRemovalError(RuntimeError):
    pass


def regions_at(regions: list[dict[str, object]], time_ms: int) -> list[dict[str, object]]:
    return [
        region for region in regions
        if bool(region.get("enabled", True)) and int(region["startMs"]) <= time_ms <= int(region["endMs"])
    ]


class SubtitleFrameCleaner:
    def __init__(self, models: SubtitleModelManager) -> None:
        self.models = models
        self._session = None

    def _session_for_lama(self):
        if self._session is not None:
            return self._session
        model_dir = self.models.models_dir()
        if not model_dir:
            raise SubtitleRemovalError("O pacote de IA para remover legendas não está instalado.")
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            options.enable_cpu_mem_arena = False
            self._session = ort.InferenceSession(str(model_dir / "lama_fp32.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise SubtitleRemovalError("O modelo LaMa não pôde ser iniciado. Reinstale o pacote de IA.") from exc
        return self._session

    @staticmethod
    def mask_for(frame, regions: list[dict[str, object]]):
        import cv2
        import numpy as np
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            x1 = max(0, min(width - 1, int(float(region["x"]) * width)))
            y1 = max(0, min(height - 1, int(float(region["y"]) * height)))
            x2 = max(x1 + 1, min(width, int(round((float(region["x"]) + float(region["width"])) * width))))
            y2 = max(y1 + 1, min(height, int(round((float(region["y"]) + float(region["height"])) * height))))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        kernel = max(3, int(round(min(height, width) * 0.004)) | 1)
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))

    def clean(self, frame, regions: list[dict[str, object]], previous_input, previous_clean):
        import cv2
        import numpy as np
        if not regions:
            return frame.copy()
        mask = self.mask_for(frame, regions)
        candidate = None
        if previous_input is not None and previous_clean is not None and previous_input.shape == frame.shape:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_input, cv2.COLOR_BGR2GRAY)
            scene_change = float(np.mean(cv2.absdiff(current_gray, previous_gray)))
            if scene_change < 30:
                flow = cv2.calcOpticalFlowFarneback(current_gray, previous_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                grid_x, grid_y = np.meshgrid(np.arange(frame.shape[1], dtype=np.float32), np.arange(frame.shape[0], dtype=np.float32))
                candidate = cv2.remap(previous_clean, grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if candidate is None:
            candidate = self._lama(frame, mask)
        feather_size = max(3, int(round(min(frame.shape[:2]) * 0.009)) | 1)
        alpha = cv2.GaussianBlur(mask, (feather_size, feather_size), 0).astype(np.float32)[..., None] / 255.0
        return np.clip(frame.astype(np.float32) * (1 - alpha) + candidate.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def _lama(self, frame, mask):
        import cv2
        import numpy as np
        points = cv2.findNonZero(mask)
        if points is None:
            return frame.copy()
        x, y, width, height = cv2.boundingRect(points)
        context = max(48, int(max(width, height) * 0.55))
        x1, y1 = max(0, x - context), max(0, y - context)
        x2, y2 = min(frame.shape[1], x + width + context), min(frame.shape[0], y + height + context)
        crop = frame[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        if crop.size == 0:
            raise SubtitleRemovalError("A região de legenda não pôde ser preparada para a IA.")
        image_512 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        image_input = np.transpose(image_512.astype(np.float32) / 255.0, (2, 0, 1))[None]
        mask_input = (mask_512.astype(np.float32) / 255.0)[None, None]
        session = self._session_for_lama()
        inputs = session.get_inputs()
        names = {item.name.lower(): item.name for item in inputs}
        image_name = names.get("image", inputs[0].name)
        mask_name = names.get("mask", inputs[1].name if len(inputs) > 1 else inputs[0].name)
        try:
            output = session.run(None, {image_name: image_input, mask_name: mask_input})[0]
        except Exception as exc:
            raise SubtitleRemovalError("O modelo LaMa falhou ao reconstruir o fundo. Nenhum borrão ou corte foi aplicado.") from exc
        result = np.asarray(output)[0]
        if result.shape[0] == 3:
            result = np.transpose(result, (1, 2, 0))
        if float(np.nanmax(result)) <= 1.5:
            result = result * 255.0
        result = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        candidate = frame.copy()
        candidate[y1:y2, x1:x2] = result
        return candidate


def remove_burned_subtitles(
    source: Path,
    destination: Path,
    probe: dict[str, object],
    regions: list[dict[str, object]],
    models: SubtitleModelManager,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, int]:
    if bool(probe.get("hdr")):
        raise SubtitleRemovalError("Vídeos HDR ainda não são suportados na remoção de legenda gravada, pois a conversão poderia alterar as cores.")
    if not regions:
        raise SubtitleRemovalError("Nenhuma região ativa foi definida para remover da imagem.")
    if int(probe["width"]) % 2 or int(probe["height"]) % 2:
        raise SubtitleRemovalError("A resolução precisa ter largura e altura pares para manter H.264 em yuv420p.")
    import cv2
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SubtitleRemovalError("O vídeo não pôde ser aberto para remover as legendas.")
    fps = float(probe["fps"])
    width, height = int(probe["width"]), int(probe["height"])
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or round(float(probe["duration"]) * fps))
    command = [
        find_binary("ffmpeg"), "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:.8f}", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-movflags", "+faststart", str(destination),
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
    cleaner = SubtitleFrameCleaner(models)
    previous_input = None
    previous_clean = None
    processed = 0
    try:
        assert process.stdin is not None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_ms = int(round(processed / fps * 1000))
            active = regions_at(regions, time_ms)
            cleaned = cleaner.clean(frame, active, previous_input, previous_clean) if active else frame
            try:
                process.stdin.write(cleaned.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise SubtitleRemovalError("O codificador H.264 parou durante a remoção das legendas.") from exc
            previous_input = frame
            previous_clean = cleaned
            processed += 1
            if progress and (processed % max(1, int(fps // 2)) == 0 or processed == total_frames):
                progress(min(1.0, processed / max(1, total_frames)), f"Reconstruindo quadro {processed} de {total_frames}")
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait(timeout=120)
        if return_code != 0 or not destination.is_file() or destination.stat().st_size == 0:
            detail = next((line for line in reversed(stderr.splitlines()) if "error" in line.lower()), "")
            raise SubtitleRemovalError(f"O vídeo sem legendas não pôde ser codificado em H.264. {detail}".strip())
    except Exception:
        if process.poll() is None:
            process.kill()
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    return {"frames": processed, "expectedFrames": total_frames}
