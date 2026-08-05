from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .azure import synthesize_azure
from .config import Settings, application_data_dir, temp_root
from .credentials import get_azure_credentials
from .media import (
    ffmpeg_command,
    mux_clean_video_command,
    mux_clean_video_with_voice_command,
    probe_video,
    strip_subtitle_tracks_command,
    unique_output,
)
from .piper_local import synthesize_piper
from .subtitle_processing import remove_burned_subtitles, remove_burned_subtitles_auto


@dataclass
class UploadRecord:
    id: str
    path: Path
    filename: str
    size: int
    duration: float
    audio_tracks: list[dict[str, object]]
    subtitle_tracks: list[dict[str, object]]
    probe: dict[str, object]

    def public(self) -> dict[str, object]:
        return {
            "uploadId": self.id,
            "filename": self.filename,
            "duration": self.duration,
            "size": self.size,
            "audioTracks": self.audio_tracks,
            "subtitleTracks": self.subtitle_tracks,
            "resolution": {"width": self.probe["width"], "height": self.probe["height"]},
            "fps": self.probe["fps"],
            "videoCodec": self.probe["videoCodec"],
            "variableFrameRate": self.probe["variableFrameRate"],
            "hdr": self.probe["hdr"],
            "color": {
                "pixelFormat": self.probe["pixelFormat"],
                "space": self.probe["colorSpace"],
                "transfer": self.probe["colorTransfer"],
                "primaries": self.probe["colorPrimaries"],
            },
        }


@dataclass
class JobRecord:
    id: str
    upload_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Aguardando processamento"
    output_path: Path | None = None
    error: str | None = None
    warning: str | None = None
    audio_asset_id: str | None = None
    source_name: str = ""

    def public(self) -> dict[str, object]:
        data: dict[str, object] = {"id": self.id, "status": self.status, "progress": round(self.progress, 1), "message": self.message}
        if self.source_name:
            data["sourceName"] = self.source_name
        if self.output_path:
            data["outputName"] = self.output_path.name
        if self.error:
            data["error"] = self.error
        if self.warning:
            data["warning"] = self.warning
        return data


@dataclass
class AudioRecord:
    """Áudio próprio enviado pelo usuário para entrar no lugar da voz sintetizada."""
    id: str
    path: Path
    filename: str
    size: int
    duration: float
    codec: str

    def public(self) -> dict[str, object]:
        return {"audioId": self.id, "filename": self.filename, "duration": self.duration, "size": self.size, "codec": self.codec}


class AudioStore:
    def __init__(self) -> None:
        self.directory = temp_root() / "audio"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, AudioRecord] = {}
        self._lock = threading.Lock()

    def add(self, path: Path, filename: str, size: int, probe: dict[str, object]) -> AudioRecord:
        record = AudioRecord(
            id=path.stem, path=path, filename=filename, size=size,
            duration=float(probe["duration"]), codec=str(probe["codec"]),
        )
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, audio_id: str) -> AudioRecord | None:
        with self._lock:
            return self._records.get(audio_id)

    def remove(self, audio_id: str) -> None:
        with self._lock:
            record = self._records.pop(audio_id, None)
        if record:
            record.path.unlink(missing_ok=True)

    def in_use(self, audio_id: str, jobs: dict[str, "JobRecord"]) -> bool:
        return any(job.audio_asset_id == audio_id and job.status in {"queued", "processing"} for job in jobs.values())


class UploadStore:
    def __init__(self) -> None:
        self.directory = temp_root() / "uploads"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, UploadRecord] = {}
        self._lock = threading.Lock()

    def add(self, path: Path, filename: str, size: int, probe: dict[str, object]) -> UploadRecord:
        record = UploadRecord(
            id=path.stem, path=path, filename=filename, size=size, duration=float(probe["duration"]),
            audio_tracks=list(probe["audioTracks"]), subtitle_tracks=list(probe["subtitleTracks"]), probe=dict(probe),
        )
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, upload_id: str) -> UploadRecord | None:
        with self._lock:
            return self._records.get(upload_id)

    def remove(self, upload_id: str) -> None:
        with self._lock:
            record = self._records.pop(upload_id, None)
        if record:
            record.path.unlink(missing_ok=True)


class JobManager:
    def __init__(self, uploads: UploadStore, scans: Any, models: Any, audio: AudioStore | None = None) -> None:
        self.uploads = uploads
        self.scans = scans
        self.models = models
        self.audio = audio or AudioStore()
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ofuscador-job")
        self._active_process: subprocess.Popen[str] | None = None

    def create(
        self, *, upload_id: str, mode: str = "audio", text: str = "", voice: str = "", gender: str = "Male",
        volume_percent: int = 5, audio_stream_index: int | None = None, provider: str = "piper",
        remove_embedded: bool = False, remove_burned_in: bool = False, scan_id: str | None = None,
        burned_in_mode: str = "regions", audio_asset_id: str | None = None,
    ) -> JobRecord:
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError("O envio do vídeo expirou. Escolha o arquivo novamente.")
        if mode not in {"audio", "subtitles", "audio_and_subtitles"}:
            raise ValueError("O modo de processamento escolhido não é válido.")
        if burned_in_mode not in {"regions", "auto"}:
            raise ValueError("O modo de remoção de legenda gravada escolhido não é válido.")
        if self.scans.upload_in_use(upload_id):
            raise RuntimeError("Aguarde o exame de legendas terminar antes de processar o vídeo.")
        uses_audio = mode in {"audio", "audio_and_subtitles"}
        if uses_audio:
            if audio_stream_index is None:
                raise ValueError("Escolha a faixa de áudio original do vídeo.")
            if provider == "file":
                if not audio_asset_id or not self.audio.get(audio_asset_id):
                    raise ValueError("Envie o arquivo de áudio que será misturado ao vídeo.")
            elif not text.strip() or not voice:
                raise ValueError("Preencha o texto, a voz e a faixa de áudio original.")
            if audio_stream_index not in {int(track["index"]) for track in upload.audio_tracks}:
                raise ValueError("A faixa de áudio escolhida não existe neste vídeo.")
        if mode in {"subtitles", "audio_and_subtitles"}:
            if not remove_embedded and not remove_burned_in:
                raise ValueError("Ative a remoção de uma faixa separada ou de uma legenda gravada.")
            if remove_embedded and not upload.subtitle_tracks and not remove_burned_in:
                raise ValueError("Este vídeo não possui uma faixa de legenda separada para remover.")
            if remove_burned_in:
                if bool(upload.probe.get("hdr")):
                    raise ValueError("A remoção de legenda gravada em HDR está bloqueada nesta versão para preservar as cores.")
                if burned_in_mode != "auto":
                    # Modo CAIXA: exige o exame revisado. O modo AUTOMÁTICO (tela toda menos
                    # rosto) não precisa de caixa nem de exame — vai direto ao processamento.
                    if not scan_id:
                        raise ValueError("Examine e revise as regiões da legenda gravada antes de processar.")
                    self.scans.regions_for_job(scan_id, upload_id)
        with self._lock:
            if any(job.upload_id == upload_id and job.status in {"queued", "processing"} for job in self._jobs.values()):
                raise RuntimeError("Este vídeo já está na fila de processamento.")
            # Vários vídeos podem ser enfileirados de uma vez, como o script .bat que
            # percorria a pasta inteira. O executor de um worker garante que só um roda
            # por vez, então a fila é atendida em ordem sem concorrer por CPU nem disco.
            job = JobRecord(id=uuid.uuid4().hex, upload_id=upload_id, audio_asset_id=audio_asset_id, source_name=upload.filename)
            self._jobs[job.id] = job
        self._executor.submit(
            self._run, job, mode, text, voice, gender, volume_percent, audio_stream_index, provider,
            remove_embedded, remove_burned_in, scan_id, burned_in_mode,
        )
        return job

    def queue_length(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status in {"queued", "processing"})

    def snapshot(self) -> dict[str, JobRecord]:
        with self._lock:
            return dict(self._jobs)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def upload_in_use(self, upload_id: str) -> bool:
        with self._lock:
            return any(job.upload_id == upload_id and job.status in {"queued", "processing"} for job in self._jobs.values())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status in {"queued", "processing"}:
                raise RuntimeError("Aguarde o processamento terminar antes de excluir.")
            self._jobs.pop(job_id, None)
        if job.output_path:
            job.output_path.unlink(missing_ok=True)
        return True

    def _update(self, job: JobRecord, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)

    def _log(self, job: JobRecord, text: str) -> None:
        """Registra o passo a passo em Dados/logs/job-<id>.txt.

        Sem isso um erro no fim do processamento não diz nada sobre o que aconteceu
        durante os quinze minutos anteriores.
        """
        try:
            directory = application_data_dir() / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"job-{job.id}.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"{text}\n")
        except OSError:
            pass

    def _synthesize(self, text: str, voice: str, provider: str, speech: Path) -> None:
        if provider == "file":
            raise RuntimeError("O áudio próprio não é sintetizado.")
        if provider == "piper":
            synthesize_piper(text, voice, speech)
        elif provider == "azure":
            try:
                key, region = get_azure_credentials()
            except Exception:
                key, region = None, None
            if not key or not region:
                raise RuntimeError("Cadastre a chave e a região Azure nas configurações.")
            synthesize_azure(text, voice, key, region, speech)
        else:
            raise RuntimeError("O provedor de voz escolhido não é válido.")

    def _run_ffmpeg(self, job: JobRecord, command: list[str], duration: float, start: float, end: float, message: str) -> None:
        self._log(job, f"[ffmpeg] {subprocess.list2cmdline(command)}")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)
        self._active_process = process
        recent: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            recent.append(line)
            recent = recent[-60:]
            if line.startswith("out_time_us="):
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                    self._update(job, progress=start + min(1.0, seconds / max(duration, 0.001)) * (end - start), message=message)
                except ValueError:
                    pass
        return_code = process.wait()
        self._active_process = None
        if return_code != 0:
            self._log(job, "[ffmpeg] saída " + str(return_code) + "\n" + "\n".join(recent))
            details = next((line for line in reversed(recent) if "error" in line.lower() or "invalid" in line.lower()), "")
            raise RuntimeError(f"O FFmpeg não conseguiu concluir o vídeo. {details}".strip())

    def _run(
        self, job: JobRecord, mode: str, text: str, voice: str, gender: str, volume_percent: int,
        audio_stream_index: int | None, provider: str, remove_embedded: bool, remove_burned_in: bool, scan_id: str | None,
        burned_in_mode: str = "regions",
    ) -> None:
        upload = self.uploads.get(job.upload_id)
        if not upload:
            self._update(job, status="failed", error="O arquivo temporário do vídeo não foi encontrado.", message="Falha")
            return
        work = temp_root() / "jobs" / job.id
        speech = work / "speech.wav"
        clean_video = work / "sem-legenda-imagem.mp4"
        partial: Path | None = None
        uses_audio = mode in {"audio", "audio_and_subtitles"}
        subtitles_active = mode in {"subtitles", "audio_and_subtitles"}
        stats: dict[str, float] | None = None
        try:
            work.mkdir(parents=True, exist_ok=False)
            self._update(job, status="processing", progress=2, message="Preparando o processamento")
            self._log(job, f"[origem] {upload.filename}")
            self._log(
                job,
                f"[origem] contêiner {upload.duration:.3f}s · vídeo {float(upload.probe['videoDuration']):.3f}s · "
                f"{upload.probe['frameCount']} quadros · {upload.probe['frameRateRational']} fps"
                f"{' · VFR' if upload.probe['variableFrameRate'] else ''}"
            )
            self._log(
                job,
                f"[origem] {upload.probe['videoCodec']} {upload.probe['width']}x{upload.probe['height']} · "
                f"{upload.probe['pixelFormat']} · espaço {upload.probe['colorSpace'] or '—'} · "
                f"transfer {upload.probe['colorTransfer'] or '—'}{' · HDR' if upload.probe['hdr'] else ''} · "
                f"{len(upload.audio_tracks)} faixa(s) de áudio · {len(upload.subtitle_tracks)} de legenda"
            )
            self._log(job, f"[modo] {mode} · faixas separadas={remove_embedded} · gravada={remove_burned_in} ({burned_in_mode})")
            if uses_audio:
                if provider == "file":
                    asset = self.audio.get(job.audio_asset_id or "")
                    if not asset or not asset.path.is_file():
                        raise RuntimeError("O arquivo de áudio enviado expirou. Envie-o novamente.")
                    # O FFmpeg lê MP3/WAV/M4A direto e o `-stream_loop -1` já repete o
                    # áudio até o vídeo acabar; converter para WAV aqui só gastaria disco.
                    speech = asset.path
                    self._update(job, progress=4, message="Preparando o áudio enviado")
                    self._log(job, f"[áudio] {asset.filename} · {asset.codec} · {asset.duration:.3f}s")
                else:
                    self._update(job, progress=4, message="Preparando a voz")
                    self._synthesize(text, voice, provider, speech)

            settings = Settings.load()
            output_dir = settings.ensure_output()
            required = upload.size * (3 if remove_burned_in else 1) + 512 * 1024 * 1024
            if shutil.disk_usage(output_dir).free < required:
                raise RuntimeError("Não há espaço livre suficiente na pasta de saída.")
            final_output = unique_output(output_dir, upload.filename, subtitles=subtitles_active)
            partial = output_dir / f".{job.id}.incompleto.mp4"

            if mode == "audio":
                assert audio_stream_index is not None
                command = ffmpeg_command(upload.path, speech, partial, audio_stream_index, upload.duration, volume_percent)
                self._update(job, progress=14, message="Combinando vídeo e áudio")
                self._run_ffmpeg(job, command, upload.duration, 14, 96, "Combinando vídeo e áudio")
            elif remove_burned_in:
                self._update(job, progress=10 if uses_audio else 4, message="Reconstruindo o fundo sob as legendas")
                lower = 12 if uses_audio else 5
                upper = 82
                report = lambda fraction, message: self._update(job, progress=lower + fraction * (upper - lower), message=message)
                if burned_in_mode == "auto":
                    # Receita AUTOMÁTICA (tela toda menos rosto), a mesma validada na Colab.
                    stats = remove_burned_subtitles_auto(
                        upload.path, clean_video, upload.probe, self.models, report,
                    )
                else:
                    assert scan_id is not None
                    regions = self.scans.regions_for_job(scan_id, upload.id)
                    stats = remove_burned_subtitles(
                        upload.path, clean_video, upload.probe, regions, self.models, report,
                    )
                self._log(
                    job,
                    f"[reconstrução] {int(stats['frames'])} de {int(stats['expectedFrames'])} quadros · "
                    f"saída {stats['outputDuration']:.3f}s · origem {stats['sourceVideoDuration']:.3f}s"
                )
                if uses_audio:
                    assert audio_stream_index is not None
                    command = mux_clean_video_with_voice_command(clean_video, upload.path, speech, partial, audio_stream_index, upload.duration, volume_percent, remove_subtitles=remove_embedded)
                    message = "Aplicando voz e preservando a duração"
                else:
                    command = mux_clean_video_command(clean_video, upload.path, partial, upload.duration, remove_subtitles=remove_embedded)
                    message = "Preservando as faixas de áudio originais"
                self._run_ffmpeg(job, command, upload.duration, 83, 97, message)
            elif mode == "subtitles":
                command = strip_subtitle_tracks_command(upload.path, partial, upload.duration)
                self._update(job, progress=14, message="Removendo faixas de legenda sem recodificar")
                self._run_ffmpeg(job, command, upload.duration, 14, 97, "Removendo faixas de legenda sem recodificar")
            else:
                assert audio_stream_index is not None
                command = ffmpeg_command(upload.path, speech, partial, audio_stream_index, upload.duration, volume_percent, keep_subtitles=not remove_embedded)
                self._update(job, progress=14, message="Removendo legendas e aplicando a voz")
                self._run_ffmpeg(job, command, upload.duration, 14, 97, "Removendo legendas e aplicando a voz")

            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("O arquivo final não foi criado corretamente.")
            result_probe = probe_video(partial)
            self._log(
                job,
                f"[resultado] contêiner {float(result_probe['duration']):.3f}s · "
                f"vídeo {float(result_probe['videoDuration']):.3f}s · {result_probe['frameCount']} quadros · "
                f"{result_probe['videoCodec']} · {len(result_probe['audioTracks'])} faixa(s) de áudio"
            )
            if remove_embedded and result_probe["subtitleTracks"]:
                raise RuntimeError("A verificação encontrou uma faixa de legenda que deveria ter sido removida.")
            # Compara duração de vídeo com duração de vídeo. `duration` do contêiner MP4
            # costuma ser a da faixa de áudio (padding do AAC) e não serve de referência.
            expected = float(stats["outputDuration"]) if stats else float(upload.probe["videoDuration"])
            obtained = float(result_probe["videoDuration"])
            tolerance = max(2.0 / float(upload.probe["fps"]), 0.15)
            difference = abs(obtained - expected)
            if difference > tolerance:
                # Aviso, não falha: um desvio de décimos de segundo não justifica jogar
                # fora o vídeo inteiro depois de reconstruir milhares de quadros.
                warning = (
                    f"A duração ficou {difference:.3f}s fora do esperado "
                    f"(obtida {obtained:.3f}s, esperada {expected:.3f}s, tolerância {tolerance:.3f}s). "
                    "O vídeo foi salvo mesmo assim — confira o fim do arquivo."
                )
                self._log(job, f"[aviso] {warning}")
                self._update(job, warning=warning)
            partial.replace(final_output)
            partial = None
            self._log(job, f"[fim] salvo em {final_output}")
            self._update(job, status="completed", progress=100, message="Vídeo pronto", output_path=final_output)
        except Exception as exc:
            detail = str(exc) or "Erro inesperado."
            self._log(job, f"[falha] {detail}")
            # Preserva o que já foi reconstruído: apagar o arquivo aqui significava
            # perder o upload, a análise OCR e todo o tempo do LaMa de uma vez.
            if partial and partial.is_file() and partial.stat().st_size > 0:
                try:
                    kept = unique_output(partial.parent, upload.filename, suffix="-com-aviso")
                    partial.replace(kept)
                    partial = None
                    detail = f"{detail} O material já processado foi mantido em {kept.name}."
                    self._log(job, f"[falha] arquivo preservado em {kept}")
                except OSError:
                    pass
            self._update(job, status="failed", message="Falha no processamento", error=detail)
        finally:
            self._active_process = None
            if partial:
                partial.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)
            # Só descarta o envio quando deu certo. Em caso de falha o vídeo e a análise
            # de legendas continuam válidos e o usuário reprocessa sem enviar tudo de novo.
            if job.status == "completed":
                self.uploads.remove(job.upload_id)


def cleanup_temporary_files() -> None:
    root = temp_root()
    for name in ("uploads", "jobs", "previews", "subtitle-previews", "audio"):
        target = root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
