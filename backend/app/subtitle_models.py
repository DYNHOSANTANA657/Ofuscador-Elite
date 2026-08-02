from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from .config import application_data_dir, resource_root, temp_root


MODEL_VERSION = "1"
MODEL_FILES = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    "lama_fp32.onnx": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
}
LAMA_URL = "https://huggingface.co/sapienkit/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MAX_PACKAGE_BYTES = 1_500_000_000


def publication_value(name: str) -> str:
    environment = os.environ.get(name, "").strip()
    if environment:
        return environment
    try:
        data = json.loads((resource_root() / "publication.json").read_text(encoding="utf-8"))
        key = "subtitleModelUrl" if name.endswith("URL") else "subtitleModelSha256"
        return str(data.get(key, "")).strip()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> str:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("O pacote contém um caminho de arquivo inseguro.")
    if len(pure.parts) != 1 or pure.name not in {*MODEL_FILES, "manifest.json"}:
        raise ValueError(f"O pacote contém um arquivo inesperado: {normalized}")
    return pure.name


class SubtitleModelManager:
    def __init__(self) -> None:
        self.root = application_data_dir() / "IA-Legendas"
        self.packages = self.root / "packages"
        self.packages.mkdir(parents=True, exist_ok=True)
        self.active_file = self.root / "active.json"
        self._lock = threading.Lock()
        self._status = "ready" if self.models_dir() else "not_installed"
        self._progress = 100.0 if self._status == "ready" else 0.0
        self._message = "Pacote de IA pronto" if self._status == "ready" else "Pacote de IA ainda não instalado"
        self._error: str | None = None

    def models_dir(self) -> Path | None:
        try:
            data = json.loads(self.active_file.read_text(encoding="utf-8"))
            package = self.packages / str(data["package"])
            if data.get("version") != MODEL_VERSION or not package.is_dir():
                return None
            if all((package / name).is_file() for name in MODEL_FILES):
                return package
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return None

    def public(self) -> dict[str, object]:
        with self._lock:
            installed = self.models_dir() is not None
            return {
                "installed": installed,
                "version": MODEL_VERSION if installed else None,
                "status": self._status,
                "progress": round(self._progress, 1),
                "message": self._message,
                "error": self._error,
                "downloadConfigured": True,
            }

    def start_install(self) -> None:
        with self._lock:
            if self._status in {"downloading", "installing"}:
                raise RuntimeError("A instalação do pacote de IA já está em andamento.")
            self._status = "downloading"
            self._progress = 1
            self._message = "Preparando o pacote de IA"
            self._error = None
        threading.Thread(target=self._install_online, name="ofuscador-ia-install", daemon=True).start()

    def install_import(self, zip_path: Path) -> None:
        with self._lock:
            if self._status in {"downloading", "installing"}:
                raise RuntimeError("A instalação do pacote de IA já está em andamento.")
        self._set("installing", 2, "Verificando o pacote de IA importado")
        try:
            self.install_zip(zip_path)
        except Exception as exc:
            self._set("failed", 0, "A importação do pacote de IA falhou", str(exc) or "Falha inesperada.")
            raise

    def _set(self, status: str, progress: float, message: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._progress = progress
            self._message = message
            self._error = error

    def _install_online(self) -> None:
        work = temp_root() / f"subtitle-model-install-{uuid.uuid4().hex}"
        try:
            work.mkdir(parents=True, exist_ok=False)
            package_url = publication_value("OFUSCADOR_SUBTITLE_MODEL_URL")
            package_hash = publication_value("OFUSCADOR_SUBTITLE_MODEL_SHA256").lower()
            if package_url:
                if len(package_hash) != 64:
                    raise RuntimeError("A publicação do pacote de IA não informou um SHA-256 válido.")
                zip_path = work / "OfuscadorElite-IA-Legendas-v1.zip"
                self._download(package_url, zip_path, package_hash, 5, 82)
                self.install_zip(zip_path)
                return

            staging = work / "package"
            staging.mkdir()
            self._copy_bundled_ocr(staging)
            self._set("downloading", 18, "Baixando o modelo LaMa local")
            self._download(LAMA_URL, staging / "lama_fp32.onnx", MODEL_FILES["lama_fp32.onnx"], 18, 85)
            self._activate(staging)
        except Exception as exc:
            self._set("failed", 0, "A instalação da IA falhou", str(exc) or "Falha inesperada.")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _download(self, url: str, destination: Path, expected_hash: str, start: float, end: float) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "OfuscadorElite/1.3"})
        digest = hashlib.sha256()
        total_read = 0
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("xb") as output:
            total = int(response.headers.get("Content-Length", 0) or 0)
            while chunk := response.read(1024 * 1024):
                total_read += len(chunk)
                if total_read > MAX_PACKAGE_BYTES:
                    raise RuntimeError("O pacote de IA ultrapassa o tamanho permitido.")
                output.write(chunk)
                digest.update(chunk)
                if total:
                    progress = start + min(1.0, total_read / total) * (end - start)
                    self._set("downloading", progress, "Baixando e verificando o pacote de IA")
        if digest.hexdigest().lower() != expected_hash.lower():
            destination.unlink(missing_ok=True)
            raise RuntimeError("O SHA-256 do pacote de IA não confere. A instalação foi cancelada.")

    def _copy_bundled_ocr(self, destination: Path) -> None:
        try:
            import rapidocr
            source = Path(rapidocr.__file__).resolve().parent / "models"
        except Exception as exc:
            raise RuntimeError("Os componentes do RapidOCR não foram encontrados no aplicativo.") from exc
        for name in MODEL_FILES:
            if name == "lama_fp32.onnx":
                continue
            candidate = source / name
            if not candidate.is_file() or sha256_file(candidate) != MODEL_FILES[name]:
                raise RuntimeError(f"O componente OCR {name} falhou na verificação SHA-256.")
            shutil.copy2(candidate, destination / name)
        self._set("downloading", 17, "Modelos RapidOCR verificados")

    def install_zip(self, zip_path: Path) -> None:
        if not zip_path.is_file() or zip_path.stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError("O pacote de IA está vazio ou ultrapassa o tamanho permitido.")
        work = temp_root() / f"subtitle-model-import-{uuid.uuid4().hex}"
        work.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                if len(archive.infolist()) > len(MODEL_FILES) + 1:
                    raise ValueError("O pacote de IA contém arquivos demais.")
                names: dict[str, zipfile.ZipInfo] = {}
                total = 0
                for info in archive.infolist():
                    name = _safe_member(info.filename)
                    if name in names or info.is_dir() or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                        raise ValueError("O pacote de IA contém uma entrada inválida ou duplicada.")
                    total += info.file_size
                    if total > MAX_PACKAGE_BYTES:
                        raise ValueError("O conteúdo do pacote de IA ultrapassa o tamanho permitido.")
                    names[name] = info
                if "manifest.json" not in names or not set(MODEL_FILES).issubset(names):
                    raise ValueError("O pacote de IA está incompleto.")
                manifest = json.loads(archive.read(names["manifest.json"]).decode("utf-8"))
                if str(manifest.get("version")) != MODEL_VERSION:
                    raise ValueError("A versão do pacote de IA não é compatível com este aplicativo.")
                declared = {str(item["name"]): str(item["sha256"]).lower() for item in manifest.get("files", [])}
                for index, (name, expected) in enumerate(MODEL_FILES.items(), 1):
                    destination = work / name
                    with archive.open(names[name]) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                    if declared.get(name) != expected or sha256_file(destination) != expected:
                        raise ValueError(f"O SHA-256 do modelo {name} não confere.")
                    self._set("installing", 82 + index * 3, f"Verificando {name}")
            self._activate(work)
            work = Path()
        finally:
            if work and work != Path():
                shutil.rmtree(work, ignore_errors=True)

    def _activate(self, staging: Path) -> None:
        self._set("installing", 92, "Ativando o pacote de IA")
        for name, expected in MODEL_FILES.items():
            candidate = staging / name
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise RuntimeError(f"O modelo {name} falhou na verificação final.")
        package_id = f"v{MODEL_VERSION}-{uuid.uuid4().hex}"
        destination = self.packages / package_id
        staging.replace(destination)
        pointer = self.root / f"active-{uuid.uuid4().hex}.json"
        pointer.write_text(json.dumps({"version": MODEL_VERSION, "package": package_id}), encoding="utf-8")
        pointer.replace(self.active_file)
        self._set("ready", 100, "Pacote de IA pronto")


SUBTITLE_MODELS = SubtitleModelManager()
