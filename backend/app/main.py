from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from . import __version__
from .azure import AzureSpeechError, VoiceCache, synthesize_azure
from .config import AUDIO_EXTENSIONS, MAX_AUDIO_BYTES, MAX_TEXT_LENGTH, MAX_VIDEO_BYTES, Settings, resource_root, temp_root
from .credentials import delete_azure_credentials, get_azure_credentials, set_azure_credentials
from .jobs import AudioStore, JobManager, UploadStore, cleanup_temporary_files
from .media import MediaError, probe_audio, probe_video
from .piper_local import PIPER_VOICES, PiperSpeechError, piper_available, synthesize_piper
from .security import COOKIE_NAME, SessionSecurity, require_local
from .subtitle_models import SUBTITLE_MODELS
from .subtitles import SubtitleScanManager

NETWORK_MODE = os.environ.get("OFUSCADOR_NETWORK_MODE") == "1"
SECURITY = SessionSecurity(NETWORK_MODE, os.environ.get("OFUSCADOR_PIN"))
VOICE_CACHE = VoiceCache()

cleanup_temporary_files()
UPLOADS = UploadStore()
AUDIO_ASSETS = AudioStore()
SUBTITLE_SCANS = SubtitleScanManager(UPLOADS, SUBTITLE_MODELS)
JOBS = JobManager(UPLOADS, SUBTITLE_SCANS, SUBTITLE_MODELS, AUDIO_ASSETS)

app = FastAPI(title="Ofuscador Elite", version=__version__, docs_url=None, redoc_url=None)


class SessionRequest(BaseModel):
    pin: str = Field(min_length=6, max_length=6, pattern="^[0-9]{6}$")


class AzureConfigRequest(BaseModel):
    key: str | None = None
    region: str | None = None
    outputDirectory: str | None = None


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    voice: str = Field(min_length=1, max_length=160)
    gender: str = Field(pattern="^(Male|Female)$")
    provider: str = Field(pattern="^(piper|azure)$")


class SubtitleRegionRequest(BaseModel):
    id: str | None = None
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    startMs: int = Field(ge=0)
    endMs: int = Field(gt=0)
    source: str = Field(default="manual", pattern="^(automatic|manual)$")
    enabled: bool = True
    text: str = Field(default="", max_length=240)
    confidence: float = Field(default=1.0, ge=0, le=1)


class SubtitleRegionsRequest(BaseModel):
    regions: list[SubtitleRegionRequest] = Field(max_length=500)


class SubtitleScanRequest(BaseModel):
    fullScreen: bool = False


class SubtitleRemovalRequest(BaseModel):
    removeEmbedded: bool = False
    removeBurnedIn: bool = False
    # "regions" = apaga só dentro das caixas revisadas (exige scanId);
    # "auto" = tela toda menos rosto (receita da Colab), sem exame nem caixa.
    burnedInMode: str = Field(default="regions", pattern="^(regions|auto)$")
    scanId: str | None = Field(default=None, min_length=32, max_length=32)


class JobRequest(BaseModel):
    uploadId: str = Field(min_length=32, max_length=32)
    mode: str = Field(default="audio", pattern="^(audio|subtitles|audio_and_subtitles)$")
    text: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    voice: str = Field(default="", max_length=160)
    gender: str = Field(default="Male", pattern="^(Male|Female)$")
    volumePercent: int = Field(default=5, ge=0, le=25)
    audioStreamIndex: int | None = Field(default=None, ge=0)
    provider: str = Field(default="piper", pattern="^(piper|azure|file)$")
    audioAssetId: str | None = Field(default=None, min_length=32, max_length=32)
    subtitleRemoval: SubtitleRemovalRequest = Field(default_factory=SubtitleRemovalRequest)

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "JobRequest":
        if self.mode in {"audio", "audio_and_subtitles"}:
            if self.audioStreamIndex is None:
                raise ValueError("Escolha a faixa de áudio original do vídeo.")
            if self.provider == "file":
                if not self.audioAssetId:
                    raise ValueError("Envie o arquivo de áudio que será misturado ao vídeo.")
            elif not self.text.strip() or not self.voice:
                raise ValueError("Preencha o texto, a voz e a faixa de áudio original.")
        if self.mode in {"subtitles", "audio_and_subtitles"}:
            removal = self.subtitleRemoval
            if not removal.removeEmbedded and not removal.removeBurnedIn:
                raise ValueError("Ative pelo menos um tipo de remoção de legenda.")
            if removal.removeBurnedIn and removal.burnedInMode != "auto" and not removal.scanId:
                raise ValueError("Examine e revise a legenda gravada antes de processar.")
        return self


def credentials() -> tuple[str | None, str | None]:
    try:
        return get_azure_credentials()
    except Exception:
        return None, None


@app.middleware("http")
async def protect_network_requests(request: Request, call_next):
    if request.url.path == "/api/session" or not request.url.path.startswith("/api/"):
        return await call_next(request)
    if not SECURITY.request_allowed(request):
        return JSONResponse(status_code=401, content={"detail": "Digite o PIN exibido no computador que está executando o aplicativo."})
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": __version__, "networkMode": NETWORK_MODE}


@app.post("/api/session")
def create_session(payload: SessionRequest, response: Response, request: Request) -> dict[str, bool]:
    allowed, retry_after = SECURITY.authenticate(request.client.host if request.client else None, payload.pin)
    if retry_after:
        raise HTTPException(status_code=429, detail=f"Muitas tentativas. Aguarde {retry_after} segundos.", headers={"Retry-After": str(retry_after)})
    if not allowed:
        raise HTTPException(status_code=401, detail="PIN incorreto.")
    response.set_cookie(COOKIE_NAME, SECURITY.issue(), httponly=True, samesite="strict", max_age=12 * 60 * 60, path="/")
    return {"authenticated": True}


@app.get("/api/config/status")
def config_status() -> dict[str, object]:
    key, region = credentials()
    settings = Settings.load()
    return {
        "credentialConfigured": bool(key and region),
        "azureConfigured": bool(key and region),
        "azureRegion": region or "",
        "piperAvailable": piper_available(),
        "defaultProvider": "piper",
        "outputDirectory": settings.output_directory,
        "networkMode": NETWORK_MODE,
        "pinRequired": NETWORK_MODE,
        "subtitleModels": SUBTITLE_MODELS.public(),
    }


@app.post("/api/config/azure")
def save_config(payload: AzureConfigRequest, request: Request) -> dict[str, bool]:
    require_local(request)
    if payload.key:
        if not payload.region or not payload.region.strip():
            raise HTTPException(status_code=422, detail="Informe também a região da conta Azure.")
        try:
            voices = VOICE_CACHE.get(payload.key.strip(), payload.region.strip().lower())
        except AzureSpeechError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not voices:
            raise HTTPException(status_code=400, detail="A conta Azure não retornou vozes pt-BR.")
        try:
            set_azure_credentials(payload.key.strip(), payload.region.strip().lower())
        except Exception as exc:
            raise HTTPException(status_code=500, detail="O cofre de credenciais do sistema não pôde salvar a chave.") from exc
    if payload.outputDirectory is not None:
        raw = payload.outputDirectory.strip()
        if not raw:
            raise HTTPException(status_code=422, detail="Informe uma pasta de saída.")
        output = Path(raw).expanduser().resolve()
        try:
            output.mkdir(parents=True, exist_ok=True)
            test = output / f".ofuscador-write-test-{uuid.uuid4().hex}"
            test.write_bytes(b"ok")
            test.unlink()
        except OSError as exc:
            raise HTTPException(status_code=400, detail="A pasta de saída não pode ser usada para gravação.") from exc
        settings = Settings(str(output))
        settings.save()
    return {"saved": True}


@app.delete("/api/config/azure")
def remove_azure_config(request: Request) -> dict[str, bool]:
    require_local(request)
    try:
        delete_azure_credentials()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="O cofre de credenciais não pôde remover a chave.") from exc
    VOICE_CACHE.clear()
    return {"deleted": True}


@app.get("/api/voices")
def voices(
    gender: str = Query(pattern="^(Male|Female)$"),
    provider: str = Query(pattern="^(piper|azure)$", default="piper"),
) -> dict[str, object]:
    if provider == "piper":
        available = PIPER_VOICES
        configured = piper_available()
    else:
        key, region = credentials()
        if not key or not region:
            return {"voices": [], "provider": provider, "configured": False}
        try:
            available = VOICE_CACHE.get(key, region)
        except AzureSpeechError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        configured = True
    return {
        "voices": [voice for voice in available if voice["gender"] == gender],
        "provider": provider,
        "configured": configured,
    }


@app.post("/api/preview")
def preview(payload: PreviewRequest) -> FileResponse:
    directory = temp_root() / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex}.wav"
    try:
        if payload.provider == "piper":
            synthesize_piper(payload.text, payload.voice, destination)
        else:
            key, region = credentials()
            if not key or not region:
                raise HTTPException(status_code=409, detail="Cadastre a chave e a região Azure nas configurações.")
            synthesize_azure(payload.text, payload.voice, key, region, destination)
    except (AzureSpeechError, PiperSpeechError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(
        destination,
        media_type="audio/wav",
        filename="previa.wav",
        headers={"Cache-Control": "no-store", "Content-Disposition": 'inline; filename="previa.wav"'},
        background=BackgroundTask(destination.unlink, missing_ok=True),
    )


@app.post("/api/uploads")
async def upload_video(video: UploadFile = File(...)) -> dict[str, object]:
    filename = video.filename or "video.mp4"
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Escolha um arquivo no formato MP4.")
    upload_id = uuid.uuid4().hex
    destination = temp_root() / "uploads" / f"{upload_id}.mp4"
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="O vídeo ultrapassa o limite de 2 GB.")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="O arquivo de vídeo está vazio.")
        probe = probe_video(destination)
        record = UPLOADS.add(destination, filename, size, probe)
        return record.public()
    except MediaError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await video.close()


@app.post("/api/audio")
async def upload_audio(audio: UploadFile = File(...)) -> dict[str, object]:
    """Recebe um MP3/WAV próprio para entrar no lugar da voz sintetizada."""
    filename = audio.filename or "audio.mp3"
    if not filename.lower().endswith(AUDIO_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Escolha um arquivo MP3, WAV, M4A, AAC, OGG ou FLAC.")
    audio_id = uuid.uuid4().hex
    suffix = Path(filename).suffix.lower()
    destination = temp_root() / "audio" / f"{audio_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="O áudio ultrapassa o limite de 300 MB.")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="O arquivo de áudio está vazio.")
        probe = probe_audio(destination)
        record = AUDIO_ASSETS.add(destination, filename, size, probe)
        return record.public()
    except MediaError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await audio.close()


@app.delete("/api/audio/{audio_id}")
def delete_audio(audio_id: str) -> dict[str, bool]:
    if len(audio_id) != 32 or not AUDIO_ASSETS.get(audio_id):
        raise HTTPException(status_code=404, detail="Áudio não encontrado.")
    if AUDIO_ASSETS.in_use(audio_id, JOBS.snapshot()):
        raise HTTPException(status_code=409, detail="Este áudio está sendo usado em um processamento.")
    AUDIO_ASSETS.remove(audio_id)
    return {"deleted": True}


@app.post("/api/jobs")
def create_job(payload: JobRequest) -> dict[str, object]:
    try:
        job = JOBS.create(
            upload_id=payload.uploadId,
            mode=payload.mode,
            text=payload.text.strip(),
            voice=payload.voice,
            gender=payload.gender,
            volume_percent=payload.volumePercent,
            audio_stream_index=payload.audioStreamIndex,
            provider=payload.provider,
            remove_embedded=payload.subtitleRemoval.removeEmbedded,
            remove_burned_in=payload.subtitleRemoval.removeBurnedIn,
            scan_id=payload.subtitleRemoval.scanId,
            burned_in_mode=payload.subtitleRemoval.burnedInMode,
            audio_asset_id=payload.audioAssetId,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.public()


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str) -> dict[str, bool]:
    if len(upload_id) != 32:
        raise HTTPException(status_code=404, detail="Envio não encontrado.")
    if not UPLOADS.get(upload_id):
        raise HTTPException(status_code=404, detail="Envio não encontrado.")
    if JOBS.upload_in_use(upload_id) or SUBTITLE_SCANS.upload_in_use(upload_id):
        raise HTTPException(status_code=409, detail="Este vídeo está sendo analisado ou processado e não pode ser trocado agora.")
    UPLOADS.remove(upload_id)
    return {"deleted": True}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabalho não encontrado.")
    return job.public()


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or job.status != "completed" or not job.output_path or not job.output_path.is_file():
        raise HTTPException(status_code=404, detail="O resultado não está disponível.")
    return FileResponse(job.output_path, media_type="video/mp4", filename=job.output_path.name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    try:
        if not JOBS.delete(job_id):
            raise HTTPException(status_code=404, detail="Trabalho não encontrado.")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True}


@app.get("/api/subtitle-model/status")
def subtitle_model_status() -> dict[str, object]:
    return SUBTITLE_MODELS.public()


@app.post("/api/subtitle-model/install")
def install_subtitle_models(request: Request) -> dict[str, object]:
    require_local(request)
    try:
        SUBTITLE_MODELS.start_install()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SUBTITLE_MODELS.public()


@app.post("/api/subtitle-model/import")
async def import_subtitle_models(request: Request, package: UploadFile = File(...)) -> dict[str, object]:
    require_local(request)
    filename = package.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="Escolha o arquivo OfuscadorElite-IA-Legendas-v1.zip.")
    destination = temp_root() / f"ia-import-{uuid.uuid4().hex}.zip"
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await package.read(1024 * 1024):
                size += len(chunk)
                if size > 1_500_000_000:
                    raise HTTPException(status_code=413, detail="O pacote de IA ultrapassa o tamanho permitido.")
                output.write(chunk)
        SUBTITLE_MODELS.install_import(destination)
        return SUBTITLE_MODELS.public()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        destination.unlink(missing_ok=True)
        await package.close()


@app.post("/api/uploads/{upload_id}/subtitle-scan")
def create_subtitle_scan(upload_id: str, payload: SubtitleScanRequest) -> dict[str, object]:
    try:
        return SUBTITLE_SCANS.create(upload_id, payload.fullScreen).public()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/subtitle-scans/{scan_id}")
def get_subtitle_scan(scan_id: str) -> dict[str, object]:
    scan = SUBTITLE_SCANS.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="A análise de legendas não foi encontrada.")
    return scan.public()


@app.put("/api/subtitle-scans/{scan_id}")
def save_subtitle_regions(scan_id: str, payload: SubtitleRegionsRequest) -> dict[str, object]:
    try:
        return SUBTITLE_SCANS.save(scan_id, [item.model_dump() for item in payload.regions]).public()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/uploads/{upload_id}/frame")
def video_review_frame(upload_id: str, timeMs: int = Query(default=0, ge=0)) -> Response:
    try:
        content = SUBTITLE_SCANS.frame_jpeg(upload_id, timeMs)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/subtitle-scans/{scan_id}/preview")
def subtitle_removal_preview(scan_id: str, timeMs: int = Query(ge=0), side: str = Query(default="after", pattern="^(before|after)$")) -> Response:
    try:
        before, after = SUBTITLE_SCANS.preview_jpegs(scan_id, timeMs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=before if side == "before" else after, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


static_directory = resource_root() / "static"
if static_directory.is_dir():
    app.mount("/", StaticFiles(directory=static_directory, html=True), name="interface")
else:
    @app.get("/")
    def missing_interface() -> dict[str, str]:
        return {"message": "A interface ainda não foi compilada."}
