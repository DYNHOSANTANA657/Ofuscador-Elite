from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

MODEL_VERSION = "1"
FILES = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    "lama_fp32.onnx": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
}
LAMA_URL = "https://huggingface.co/sapienkit/LaMa-ONNX/resolve/main/lama_fp32.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_lama(destination: Path) -> None:
    request = urllib.request.Request(LAMA_URL, headers={"User-Agent": "OfuscadorEliteBuilder/1.3"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != FILES["lama_fp32.onnx"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("O SHA-256 do LaMa baixado não confere.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o pacote de IA local do Ofuscador Elite.")
    parser.add_argument("--lama", type=Path, help="Caminho de lama_fp32.onnx já baixado.")
    parser.add_argument("--download-lama", action="store_true", help="Baixa o LaMa oficial e verifica o SHA-256.")
    parser.add_argument("--output", type=Path, default=Path("dist") / "OfuscadorElite-IA-Legendas-v1.zip")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} já existe e não será substituído.")
    output.parent.mkdir(parents=True, exist_ok=True)

    import rapidocr
    ocr_root = Path(rapidocr.__file__).resolve().parent / "models"
    lama = args.lama.expanduser().resolve() if args.lama else output.parent / ".lama_fp32.onnx.download"
    temporary_lama = args.lama is None
    if temporary_lama:
        if not args.download_lama:
            raise RuntimeError("Informe --lama ou use --download-lama.")
        if lama.exists():
            raise FileExistsError(f"O temporário {lama} já existe; remova-o antes de continuar.")
        download_lama(lama)
    sources = {name: (lama if name == "lama_fp32.onnx" else ocr_root / name) for name in FILES}
    try:
        manifest_files = []
        for name, expected in FILES.items():
            source = sources[name]
            if not source.is_file() or sha256(source) != expected:
                raise RuntimeError(f"O modelo {name} está ausente ou falhou na verificação SHA-256.")
            manifest_files.append({"name": name, "size": source.stat().st_size, "sha256": expected})
        manifest = {"name": "OfuscadorElite-IA-Legendas", "version": MODEL_VERSION, "components": {"ocr": "RapidOCR 3.9.1 / PP-OCRv6", "inpainting": "LaMa ONNX FP32"}, "files": manifest_files}
        partial = output.with_suffix(output.suffix + ".partial")
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for name, source in sources.items():
                archive.write(source, name)
        partial.replace(output)
        digest = sha256(output)
        sums = output.parent / "SHA256SUMS.txt"
        existing = sums.read_text(encoding="utf-8").splitlines() if sums.exists() else []
        existing = [line for line in existing if not line.endswith(f"  {output.name}")]
        existing.append(f"{digest.upper()}  {output.name}")
        sums_tmp = output.parent / ".SHA256SUMS.txt.partial"
        sums_tmp.write_text("\n".join(existing) + "\n", encoding="utf-8")
        sums_tmp.replace(sums)
        print(f"Pacote criado: {output}")
        print(f"SHA-256: {digest.upper()}")
    finally:
        if temporary_lama:
            lama.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
