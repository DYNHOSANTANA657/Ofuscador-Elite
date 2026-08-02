from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.subtitle_models import SubtitleModelManager, _safe_member
from app.subtitle_processing import SubtitleFrameCleaner, regions_at
from app.subtitles import validate_region


def test_region_validation_and_time_lookup() -> None:
    region = validate_region({"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.15, "startMs": 100, "endMs": 900, "source": "manual"}, 1000)
    assert region["source"] == "manual"
    assert len(regions_at([region], 500)) == 1
    assert regions_at([region], 950) == []
    with pytest.raises(ValueError, match="limites"):
        validate_region({"x": 0.8, "y": 0.8, "width": 0.4, "height": 0.1, "startMs": 0, "endMs": 100}, 1000)


def test_zip_path_validation_rejects_traversal() -> None:
    assert _safe_member("lama_fp32.onnx") == "lama_fp32.onnx"
    with pytest.raises(ValueError, match="inseguro"):
        _safe_member("../lama_fp32.onnx")
    with pytest.raises(ValueError, match="inesperado"):
        _safe_member("models/evil.onnx")


def test_model_import_rejects_malicious_zip(tmp_path: Path) -> None:
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"version": "1", "files": []}))
        archive.writestr("../escape.onnx", b"bad")
    manager = SubtitleModelManager()
    with pytest.raises(ValueError, match="inseguro"):
        manager.install_zip(package)
    assert not (tmp_path.parent / "escape.onnx").exists()


class NoModelNeeded:
    def models_dir(self):
        return None


def test_temporal_recovery_changes_only_mask_area() -> None:
    import cv2
    height, width = 90, 160
    background = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    current = background.copy()
    cv2.putText(current, "TESTE", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    region = {"x": 0.2, "y": 0.55, "width": 0.65, "height": 0.35, "startMs": 0, "endMs": 1000, "enabled": True}
    cleaned = SubtitleFrameCleaner(NoModelNeeded()).clean(current, [region], background, background)
    mask = SubtitleFrameCleaner.mask_for(current, [region])
    assert np.array_equal(cleaned[mask == 0], current[mask == 0])
    assert float(np.mean(np.abs(cleaned[mask > 0].astype(float) - current[mask > 0].astype(float)))) > 1
