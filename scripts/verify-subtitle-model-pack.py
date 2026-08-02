from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica instalação, OCR e LaMa do pacote de IA.")
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="ofuscador-ia-verify-") as temporary:
        root = Path(temporary).resolve()
        os.environ["OFUSCADOR_DATA_DIR"] = str(root / "data")
        os.environ["OFUSCADOR_TEMP_DIR"] = str(root / "temp")
        import cv2
        import numpy as np
        from rapidocr import RapidOCR
        from app.subtitle_models import SubtitleModelManager
        from app.subtitle_processing import SubtitleFrameCleaner

        manager = SubtitleModelManager()
        manager.install_import(package)
        models = manager.models_dir()
        if not models:
            raise RuntimeError("O pacote não foi ativado.")
        frame = np.full((256, 512, 3), (80, 120, 165), dtype=np.uint8)
        cv2.putText(frame, "LEGENDA TESTE", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        ocr = RapidOCR(params={"Global.model_root_dir": str(models), "Global.log_level": "warning"})
        result = ocr(frame)
        if getattr(result, "boxes", None) is None:
            raise RuntimeError("O RapidOCR não detectou o texto de diagnóstico.")
        region = {"x": 0.14, "y": 0.68, "width": 0.74, "height": 0.25, "startMs": 0, "endMs": 1000, "enabled": True}
        cleaned = SubtitleFrameCleaner(manager).clean(frame, [region], None, None)
        if cleaned.shape != frame.shape or np.array_equal(cleaned, frame):
            raise RuntimeError("O LaMa não produziu uma reconstrução válida.")
        print(json.dumps({"installed": True, "ocr": True, "lama": True, "shape": list(cleaned.shape)}))


if __name__ == "__main__":
    main()
