from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .subtitle_models import SubtitleModelManager


def validate_region(raw: dict[str, Any], duration_ms: int) -> dict[str, object]:
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw["width"])
        height = float(raw["height"])
        start_ms = int(raw.get("startMs", 0))
        end_ms = int(raw.get("endMs", duration_ms))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Uma região de legenda possui coordenadas inválidas.") from exc
    if not all(0 <= value <= 1 for value in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("As coordenadas da região precisam estar entre 0 e 1.")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("Uma região de legenda ultrapassa os limites da imagem.")
    if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms + 1000:
        raise ValueError("O começo ou o fim de uma região de legenda é inválido.")
    return {
        "id": str(raw.get("id") or uuid.uuid4().hex),
        "x": round(x, 6),
        "y": round(y, 6),
        "width": round(width, 6),
        "height": round(height, 6),
        "startMs": start_ms,
        "endMs": min(end_ms, duration_ms),
        "source": "manual" if raw.get("source") == "manual" else "automatic",
        "enabled": bool(raw.get("enabled", True)),
        "text": str(raw.get("text", ""))[:240],
        "confidence": round(float(raw.get("confidence", 1.0)), 4),
    }


def _iou(a: dict[str, object], b: dict[str, object]) -> float:
    ax1, ay1 = float(a["x"]), float(a["y"])
    ax2, ay2 = ax1 + float(a["width"]), ay1 + float(a["height"])
    bx1, by1 = float(b["x"]), float(b["y"])
    bx2, by2 = bx1 + float(b["width"]), by1 + float(b["height"])
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


@dataclass
class SubtitleScan:
    id: str
    upload_id: str
    duration_ms: int
    full_screen: bool
    status: str = "queued"
    progress: float = 0.0
    message: str = "Aguardando análise"
    regions: list[dict[str, object]] = field(default_factory=list)
    sample_times_ms: list[int] = field(default_factory=list)
    error: str | None = None

    def public(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "uploadId": self.upload_id,
            "status": self.status,
            "progress": round(self.progress, 1),
            "message": self.message,
            "fullScreen": self.full_screen,
            "regions": copy.deepcopy(self.regions),
            "sampleTimesMs": list(self.sample_times_ms),
        }
        if self.error:
            result["error"] = self.error
        return result


class SubtitleScanManager:
    def __init__(self, uploads: Any, models: SubtitleModelManager) -> None:
        self.uploads = uploads
        self.models = models
        self._scans: dict[str, SubtitleScan] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ofuscador-ocr")

    def create(self, upload_id: str, full_screen: bool = False) -> SubtitleScan:
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError("O envio do vídeo expirou. Escolha o arquivo novamente.")
        if not self.models.models_dir():
            raise RuntimeError("Instale o pacote de IA local antes de examinar legendas gravadas.")
        with self._lock:
            if any(item.status in {"queued", "scanning"} for item in self._scans.values()):
                raise RuntimeError("Já existe uma análise de legendas em andamento.")
            scan = SubtitleScan(uuid.uuid4().hex, upload_id, int(round(upload.duration * 1000)), full_screen)
            self._scans[scan.id] = scan
        self._executor.submit(self._run, scan)
        return scan

    def get(self, scan_id: str) -> SubtitleScan | None:
        with self._lock:
            return self._scans.get(scan_id)

    def upload_in_use(self, upload_id: str) -> bool:
        with self._lock:
            return any(scan.upload_id == upload_id and scan.status in {"queued", "scanning"} for scan in self._scans.values())

    def save(self, scan_id: str, regions: list[dict[str, Any]]) -> SubtitleScan:
        scan = self.get(scan_id)
        if not scan:
            raise ValueError("A análise de legendas não foi encontrada.")
        if scan.status != "completed":
            raise RuntimeError("Aguarde o exame automático terminar antes de corrigir as regiões.")
        if len(regions) > 500:
            raise ValueError("O limite é de 500 regiões de legenda por vídeo.")
        checked = [validate_region(region, scan.duration_ms) for region in regions]
        with self._lock:
            scan.regions = checked
            scan.message = "Revisão salva"
        return scan

    def regions_for_job(self, scan_id: str, upload_id: str) -> list[dict[str, object]]:
        scan = self.get(scan_id)
        if not scan or scan.upload_id != upload_id or scan.status != "completed":
            raise ValueError("A análise de legendas não está pronta para este vídeo.")
        return [copy.deepcopy(region) for region in scan.regions if region.get("enabled", True)]

    def _update(self, scan: SubtitleScan, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(scan, key, value)

    def _run(self, scan: SubtitleScan) -> None:
        upload = self.uploads.get(scan.upload_id)
        if not upload:
            self._update(scan, status="failed", error="O vídeo temporário não foi encontrado.", message="Falha")
            return
        try:
            import cv2
            import numpy as np
            from rapidocr import RapidOCR

            model_dir = self.models.models_dir()
            if not model_dir:
                raise RuntimeError("O pacote de IA local não está instalado.")
            self._update(scan, status="scanning", progress=2, message="Carregando o RapidOCR")
            ocr = RapidOCR(params={"Global.model_root_dir": str(model_dir), "Global.text_score": 0.45, "Global.log_level": "warning"})
            capture = cv2.VideoCapture(str(upload.path))
            if not capture.isOpened():
                raise RuntimeError("O vídeo não pôde ser aberto para o exame de legendas.")
            fps = float(upload.probe["fps"])
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or round(upload.duration * fps))
            step = max(1, int(round(fps / 2.0)))
            total_samples = max(1, (total_frames + step - 1) // step)
            regions: list[dict[str, object]] = []
            active: list[dict[str, object]] = []
            sample_times: list[int] = []
            previous_small = None
            scene = 0
            sample_index = 0
            frame_index = 0
            while frame_index < total_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    break
                time_ms = min(scan.duration_ms, int(round(frame_index / fps * 1000)))
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54))
                if previous_small is not None and float(np.mean(cv2.absdiff(small, previous_small))) > 34:
                    scene += 1
                    active = []
                previous_small = small
                y_offset = 0 if scan.full_screen else int(frame.shape[0] * 0.52)
                target = frame[y_offset:, :] if y_offset else frame
                result = ocr(target)
                detections: list[dict[str, object]] = []
                boxes = getattr(result, "boxes", None)
                txts = getattr(result, "txts", None) or []
                scores = getattr(result, "scores", None) or []
                if boxes is not None:
                    for box, text, score in zip(boxes, txts, scores):
                        points = np.asarray(box, dtype=float)
                        x1, x2 = float(points[:, 0].min()), float(points[:, 0].max())
                        y1, y2 = float(points[:, 1].min()) + y_offset, float(points[:, 1].max()) + y_offset
                        pad_x = max(3.0, (y2 - y1) * 0.18)
                        pad_y = max(3.0, (y2 - y1) * 0.24)
                        x1, x2 = max(0.0, x1 - pad_x), min(float(frame.shape[1]), x2 + pad_x)
                        y1, y2 = max(0.0, y1 - pad_y), min(float(frame.shape[0]), y2 + pad_y)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        detections.append({
                            "x": x1 / frame.shape[1], "y": y1 / frame.shape[0],
                            "width": (x2 - x1) / frame.shape[1], "height": (y2 - y1) / frame.shape[0],
                            "text": str(text), "confidence": float(score), "scene": scene,
                        })
                for detection in detections:
                    match = max(
                        (item for item in active if item.get("scene") == scene and time_ms - int(item["endMs"]) <= 900),
                        key=lambda item: _iou(item, detection),
                        default=None,
                    )
                    if match is not None and _iou(match, detection) >= 0.2:
                        x1 = min(float(match["x"]), float(detection["x"]))
                        y1 = min(float(match["y"]), float(detection["y"]))
                        x2 = max(float(match["x"]) + float(match["width"]), float(detection["x"]) + float(detection["width"]))
                        y2 = max(float(match["y"]) + float(match["height"]), float(detection["y"]) + float(detection["height"]))
                        match.update(x=x1, y=y1, width=x2 - x1, height=y2 - y1, endMs=min(scan.duration_ms, time_ms + 550))
                        if str(detection["text"]) not in str(match.get("text", "")):
                            match["text"] = (str(match.get("text", "")) + " / " + str(detection["text"]))[:240]
                        match["confidence"] = max(float(match["confidence"]), float(detection["confidence"]))
                    else:
                        region = {
                            "id": uuid.uuid4().hex, **detection,
                            "startMs": max(0, time_ms - 300), "endMs": min(scan.duration_ms, time_ms + 550),
                            "source": "automatic", "enabled": True,
                        }
                        regions.append(region)
                        active.append(region)
                if detections and (not sample_times or time_ms - sample_times[-1] >= 1500):
                    sample_times.append(time_ms)
                sample_index += 1
                self._update(scan, progress=4 + sample_index / total_samples * 94, message=f"Examinando quadro {sample_index} de {total_samples}")
                frame_index += step
            capture.release()
            for item in regions:
                item.pop("scene", None)
            if not sample_times:
                sample_times = [min(scan.duration_ms - 1, scan.duration_ms // 2)]
            self._update(scan, status="completed", progress=100, message=f"Exame concluído: {len(regions)} região(ões)", regions=regions, sample_times_ms=sample_times[:80])
        except Exception as exc:
            self._update(scan, status="failed", progress=0, message="O exame de legendas falhou", error=str(exc) or "Erro inesperado.")

    def frame_jpeg(self, upload_id: str, time_ms: int) -> bytes:
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError("O envio do vídeo expirou.")
        import cv2
        capture = cv2.VideoCapture(str(upload.path))
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, min(time_ms, int(upload.duration * 1000) - 1)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("Não foi possível obter este quadro do vídeo.")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Não foi possível gerar o quadro de revisão.")
        return encoded.tobytes()

    def preview_jpegs(self, scan_id: str, time_ms: int) -> tuple[bytes, bytes]:
        scan = self.get(scan_id)
        if not scan or scan.status != "completed":
            raise ValueError("A análise de legendas ainda não está pronta.")
        upload = self.uploads.get(scan.upload_id)
        if not upload:
            raise ValueError("O envio do vídeo expirou.")
        import cv2
        from .subtitle_processing import SubtitleFrameCleaner

        capture = cv2.VideoCapture(str(upload.path))
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, min(time_ms, scan.duration_ms - 1)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("Não foi possível obter o quadro da prévia.")
        active = [item for item in scan.regions if item.get("enabled", True) and int(item["startMs"]) <= time_ms <= int(item["endMs"])]
        if not active:
            raise ValueError("Não existe uma região ativa neste instante. Ajuste o tempo ou os intervalos.")
        cleaner = SubtitleFrameCleaner(self.models)
        after = cleaner.clean(frame, active, None, None)
        ok_before, before_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ok_after, after_jpg = cv2.imencode(".jpg", after, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok_before or not ok_after:
            raise RuntimeError("Não foi possível gerar a prévia antes e depois.")
        return before_jpg.tobytes(), after_jpg.tobytes()
