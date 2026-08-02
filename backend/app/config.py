from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "OfuscadorElite"
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
MAX_TEXT_LENGTH = 5_000


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def application_data_dir() -> Path:
    override = os.environ.get("OFUSCADOR_DATA_DIR")
    if override:
        path = Path(override).expanduser().resolve()
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    elif is_frozen():
        path = Path(sys.executable).resolve().parent / "Dados"
    else:
        path = Path(__file__).resolve().parents[2] / ".local-data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Movies" / "OfuscadorElite" / "Saidas"
    if is_frozen():
        return Path(sys.executable).resolve().parent / "Saidas"
    return Path(__file__).resolve().parents[2] / "Saidas"


def temp_root() -> Path:
    override = os.environ.get("OFUSCADOR_TEMP_DIR")
    path = Path(override).resolve() if override else Path(tempfile.gettempdir()) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_binary(name: str) -> str:
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates = [
        resource_root() / "bin" / executable,
        Path(__file__).resolve().parents[2] / "vendor" / f"{platform.system().lower()}-{platform.machine().lower()}" / executable,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    located = shutil.which(name)
    if located:
        return located
    raise RuntimeError(f"{name} não foi encontrado no pacote do aplicativo.")


@dataclass
class Settings:
    output_directory: str

    @classmethod
    def load(cls) -> "Settings":
        path = application_data_dir() / "settings.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                configured = Path(str(data.get("output_directory", ""))).expanduser()
                if str(configured):
                    return cls(str(configured.resolve()))
            except (OSError, ValueError, TypeError):
                pass
        return cls(str(default_output_dir().resolve()))

    def save(self) -> None:
        path = application_data_dir() / "settings.json"
        path.write_text(json.dumps({"output_directory": self.output_directory}, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_output(self) -> Path:
        output = Path(self.output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        return output
