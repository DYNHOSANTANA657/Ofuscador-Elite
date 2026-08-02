from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Callable

from .config import find_binary
from .subtitle_models import SubtitleModelManager


class SubtitleRemovalError(RuntimeError):
    pass


def _read_exact(stream: IO[bytes], size: int) -> bytearray | None:
    """Lê exatamente um quadro do pipe. Devolve None no fim do vídeo.

    Um `read()` em pipe pode voltar parcial; sem este laço um quadro sairia
    deslocado e todos os seguintes ficariam corrompidos.
    """
    buffer = bytearray(size)
    view = memoryview(buffer)
    filled = 0
    while filled < size:
        received = stream.readinto(view[filled:])
        if not received:
            return None
        filled += received
    return buffer


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""


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
) -> dict[str, float]:
    if bool(probe.get("hdr")):
        raise SubtitleRemovalError("Vídeos HDR ainda não são suportados na remoção de legenda gravada, pois a conversão poderia alterar as cores.")
    if not regions:
        raise SubtitleRemovalError("Nenhuma região ativa foi definida para remover da imagem.")
    if int(probe["width"]) % 2 or int(probe["height"]) % 2:
        raise SubtitleRemovalError("A resolução precisa ter largura e altura pares para manter H.264 em yuv420p.")
    import numpy as np
    fps = float(probe["fps"])
    rate = str(probe.get("frameRateRational") or f"{fps:.8f}")
    width, height = int(probe["width"]), int(probe["height"])
    frame_bytes = width * height * 3
    video_duration = float(probe.get("videoDuration") or probe["duration"])
    total_frames = int(probe.get("frameCount") or round(video_duration * fps))

    # Decodificar com o mesmo FFmpeg que codifica: o OpenCV usa outro build de
    # libavcodec e pode devolver menos quadros em vídeos com B-frames ou GOP aberto,
    # encurtando o resultado. `-fps_mode cfr -r` também normaliza VFR na entrada,
    # então o número de quadros que entra é exatamente o que sai.
    decode_command = [
        find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "cfr", "-r", rate,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    encode_command = [
        find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-movflags", "+faststart", str(destination),
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    logs = Path(tempfile.mkdtemp(prefix="ofuscador-ffmpeg-"))
    decode_log, encode_log = logs / "decode.txt", logs / "encode.txt"
    cleaner = SubtitleFrameCleaner(models)
    previous_input = None
    previous_clean = None
    processed = 0
    decoder = encoder = None
    try:
        # stderr vai para arquivo, não para PIPE: um PIPE de erro que ninguém lê
        # enche e trava o FFmpeg no meio de um vídeo longo.
        with decode_log.open("wb") as decode_errors, encode_log.open("wb") as encode_errors:
            decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=decode_errors, creationflags=flags)
            encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_errors, creationflags=flags)
            assert decoder.stdout is not None and encoder.stdin is not None
            while True:
                raw = _read_exact(decoder.stdout, frame_bytes)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                time_ms = int(round(processed / fps * 1000))
                active = regions_at(regions, time_ms)
                cleaned = cleaner.clean(frame, active, previous_input, previous_clean) if active else frame
                try:
                    encoder.stdin.write(cleaned.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    raise SubtitleRemovalError(f"O codificador H.264 parou no quadro {processed}. {_tail(encode_log)}".strip()) from exc
                previous_input = frame
                previous_clean = cleaned
                processed += 1
                if progress and (processed % max(1, int(fps // 2)) == 0 or processed == total_frames):
                    progress(min(1.0, processed / max(1, total_frames)), f"Reconstruindo quadro {processed} de {total_frames}")
            decoder.stdout.close()
            try:
                decode_status = decoder.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise SubtitleRemovalError("A leitura dos quadros não terminou no tempo esperado.") from exc
            if decode_status != 0:
                raise SubtitleRemovalError(f"O FFmpeg não conseguiu ler todos os quadros do vídeo. {_tail(decode_log)}".strip())
            encoder.stdin.close()
            try:
                # Fechar o stdin não encerra o codificador na hora: o libx264 ainda esvazia
                # o lookahead e o `+faststart` reescreve o índice do MP4 inteiro no fim.
                encode_status = encoder.wait(timeout=1800)
            except subprocess.TimeoutExpired as exc:
                raise SubtitleRemovalError("O codificador H.264 não terminou no tempo esperado.") from exc
            if encode_status != 0 or not destination.is_file() or destination.stat().st_size == 0:
                raise SubtitleRemovalError(f"O vídeo sem legendas não pôde ser codificado em H.264. {_tail(encode_log)}".strip())
        if processed == 0:
            raise SubtitleRemovalError("Nenhum quadro foi lido do vídeo. O arquivo pode estar corrompido.")
        # Falha aqui, com números, em vez de deixar o erro aparecer só no fim do mux.
        missing = total_frames - processed
        if missing > 0 and missing / max(1, total_frames) > 0.01 and missing / fps > 0.5:
            raise SubtitleRemovalError(
                f"A leitura terminou cedo: {processed} de {total_frames} quadros "
                f"({missing / fps:.2f}s a menos). {_tail(decode_log)}".strip()
            )
    except Exception:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        destination.unlink(missing_ok=True)
        raise
    finally:
        # No Windows um arquivo ainda aberto por um processo filho não pode ser apagado;
        # deixar esse OSError escapar aqui substituiria a exceção original por outra.
        try:
            shutil.rmtree(logs, ignore_errors=True)
        except OSError:
            pass
    return {
        "frames": processed,
        "expectedFrames": total_frames,
        "outputDuration": processed / fps,
        "sourceVideoDuration": video_duration,
    }
