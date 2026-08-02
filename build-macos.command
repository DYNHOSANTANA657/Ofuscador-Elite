#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_ROOT="$PROJECT_ROOT/.build-macos"
DIST_ROOT="$PROJECT_ROOT/dist"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="1.3.3"
# Runners macOS do GitHub Actions não trazem Rosetta 2 nem um Python universal2,
# então o pacote Intel não pode ser gerado lá. Com OFUSCADOR_SKIP_INTEL=1 o
# construtor entrega só o arm64 em vez de abortar.
SKIP_INTEL="${OFUSCADOR_SKIP_INTEL:-0}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Este construtor deve ser executado em um Mac Apple Silicon."
  exit 1
fi
if [[ -e "$BUILD_ROOT" ]]; then
  echo "A pasta temporária $BUILD_ROOT já existe. Renomeie ou remova antes de reconstruir."
  exit 1
fi
zip_names=("OfuscadorElite-macOS-arm64-v$VERSION.zip")
if [[ "$SKIP_INTEL" != "1" ]]; then
  zip_names+=("OfuscadorElite-macOS-x64-v$VERSION.zip")
fi
for zip_name in "${zip_names[@]}"; do
  if [[ -e "$DIST_ROOT/$zip_name" ]]; then
    echo "$DIST_ROOT/$zip_name já existe e não será substituído."
    exit 1
  fi
done
if [[ "$SKIP_INTEL" != "1" ]] && ! /usr/bin/arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
  echo "Rosetta 2 não está disponível. Instale com: softwareupdate --install-rosetta"
  echo "Para gerar somente o pacote Apple Silicon, use OFUSCADOR_SKIP_INTEL=1."
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11)' >/dev/null; then
  echo "Python 3.11 ou superior não foi encontrado."
  exit 1
fi
PYTHON_EXEC="$($PYTHON_BIN -c 'import sys; print(sys.executable)')"
PYTHON_FILE="$(file "$PYTHON_EXEC")"
if [[ "$PYTHON_FILE" != *"arm64"* ]]; then
  echo "O Python informado não contém a arquitetura arm64."
  exit 1
fi
if [[ "$SKIP_INTEL" != "1" && "$PYTHON_FILE" != *"x86_64"* ]]; then
  echo "Use o instalador universal2 do Python 3.12, que contém arm64 e x86_64."
  echo "Para gerar somente o pacote Apple Silicon, use OFUSCADOR_SKIP_INTEL=1."
  exit 1
fi

mkdir -p "$BUILD_ROOT/downloads" "$BUILD_ROOT/bin/arm64" "$BUILD_ROOT/bin/x64" "$DIST_ROOT"
MODEL_PACKAGE="$DIST_ROOT/OfuscadorElite-IA-Legendas-v1.zip"
if [[ ! -f "$MODEL_PACKAGE" ]]; then
  echo "Gere primeiro OfuscadorElite-IA-Legendas-v1.zip com scripts/build-subtitle-model-pack.py."
  exit 1
fi
SUBTITLE_MODEL_URL="${OFUSCADOR_SUBTITLE_MODEL_URL:-}"
SUBTITLE_MODEL_SHA256="${OFUSCADOR_SUBTITLE_MODEL_SHA256:-}"
if [[ -n "$SUBTITLE_MODEL_URL" && ! "$SUBTITLE_MODEL_SHA256" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "Defina OFUSCADOR_SUBTITLE_MODEL_SHA256 com 64 caracteres hexadecimais junto da URL."
  exit 1
fi
printf '{"subtitleModelUrl":"%s","subtitleModelSha256":"%s"}\n' "$SUBTITLE_MODEL_URL" "$SUBTITLE_MODEL_SHA256" > "$BUILD_ROOT/publication.json"

if command -v pnpm >/dev/null 2>&1; then
  (cd "$PROJECT_ROOT" && pnpm run build:portable)
  (cd "$PROJECT_ROOT" && pnpm test)
  (cd "$PROJECT_ROOT" && pnpm run typecheck:portable)
elif [[ ! -f "$PROJECT_ROOT/backend/static/index.html" ]]; then
  echo "Instale Node.js e pnpm para compilar a interface."
  exit 1
fi

download_binary() {
  local url="$1" destination="$2" archive="$BUILD_ROOT/downloads/$(basename "$destination").$(basename "$(dirname "$destination")").zip"
  local entry
  /usr/bin/curl --fail --location --retry 3 --output "$archive" "$url"
  entry="$(/usr/bin/unzip -Z1 "$archive" | grep -v '/$' | head -n 1)"
  if [[ -z "$entry" ]]; then
    echo "O arquivo baixado de $url não contém um binário."
    exit 1
  fi
  /usr/bin/unzip -p "$archive" "$entry" > "$destination"
  chmod +x "$destination"
}

download_binary "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip" "$BUILD_ROOT/bin/arm64/ffmpeg"
download_binary "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip" "$BUILD_ROOT/bin/arm64/ffprobe"

file "$BUILD_ROOT/bin/arm64/ffmpeg" | grep -q arm64
file "$BUILD_ROOT/bin/arm64/ffprobe" | grep -q arm64
"$BUILD_ROOT/bin/arm64/ffmpeg" -version >/dev/null
"$BUILD_ROOT/bin/arm64/ffmpeg" -hide_banner -encoders 2>&1 | grep -q libx264

"$PYTHON_EXEC" -m venv "$BUILD_ROOT/venv-arm64"
ARM_PYTHON="$BUILD_ROOT/venv-arm64/bin/python"
"$ARM_PYTHON" -m pip install --disable-pip-version-check -r "$PROJECT_ROOT/backend/requirements-build.txt"

if [[ "$SKIP_INTEL" != "1" ]]; then
  download_binary "https://evermeet.cx/ffmpeg/getrelease/zip" "$BUILD_ROOT/bin/x64/ffmpeg"
  download_binary "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip" "$BUILD_ROOT/bin/x64/ffprobe"
  file "$BUILD_ROOT/bin/x64/ffmpeg" | grep -q x86_64
  file "$BUILD_ROOT/bin/x64/ffprobe" | grep -q x86_64
  /usr/bin/arch -x86_64 "$BUILD_ROOT/bin/x64/ffmpeg" -version >/dev/null
  /usr/bin/arch -x86_64 "$BUILD_ROOT/bin/x64/ffmpeg" -hide_banner -encoders 2>&1 | grep -q libx264
  /usr/bin/arch -x86_64 "$PYTHON_EXEC" -m venv "$BUILD_ROOT/venv-x64"
  X64_PYTHON="$BUILD_ROOT/venv-x64/bin/python"
  /usr/bin/arch -x86_64 "$X64_PYTHON" -m pip install --disable-pip-version-check -r "$PROJECT_ROOT/backend/requirements-build.txt"
fi

build_arch() {
  local arch="$1" target="$2" binary_dir="$BUILD_ROOT/bin/$1" package_dir="$BUILD_ROOT/package-$1"
  local python_command=("$ARM_PYTHON")
  if [[ "$target" == "x86_64" ]]; then
    python_command=(/usr/bin/arch -x86_64 "$X64_PYTHON")
  fi
  (cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT/backend" "${python_command[@]}" scripts/verify-subtitle-model-pack.py "$MODEL_PACKAGE")
  (cd "$PROJECT_ROOT/backend" && PATH="$binary_dir:$PATH" "${python_command[@]}" -m pytest tests -q)
  "${python_command[@]}" -m PyInstaller --noconfirm --clean --windowed --onedir --name OfuscadorElite \
    --target-architecture "$target" --paths "$PROJECT_ROOT/backend" \
    --distpath "$BUILD_ROOT/dist-$1" --workpath "$BUILD_ROOT/work-$1" --specpath "$BUILD_ROOT/spec-$1" \
    --add-data "$PROJECT_ROOT/backend/static:static" \
    --add-data "$PROJECT_ROOT/backend/voices:voices" \
    --add-data "$BUILD_ROOT/publication.json:." \
    --add-binary "$binary_dir/ffmpeg:bin" --add-binary "$binary_dir/ffprobe:bin" \
    --hidden-import keyring.backends.macOS --collect-all keyring --collect-all piper --collect-all onnxruntime --collect-all rapidocr --collect-all cv2 \
    "$PROJECT_ROOT/backend/launcher_entry.py"
  mkdir -p "$package_dir/Licencas"
  cp -R "$BUILD_ROOT/dist-$1/OfuscadorElite.app" "$package_dir/OfuscadorElite.app"
  cp "$PROJECT_ROOT/LEIA-ME.txt" "$package_dir/"
  cp "$PROJECT_ROOT/licenses/FFmpeg-GPLv3.txt" "$package_dir/Licencas/"
  cp "$PROJECT_ROOT/licenses/Piper-GPLv3.txt" "$package_dir/Licencas/"
  cp "$PROJECT_ROOT/backend/voices/MODEL_CARD-pt_BR-faber-medium.txt" "$package_dir/Licencas/Modelo-Faber-CC0.txt"
  cp "$PROJECT_ROOT/licenses/AVISOS-DE-TERCEIROS.txt" "$package_dir/Licencas/"
  cp "$PROJECT_ROOT/licenses/IA-Legendas-APACHE-MIT.txt" "$package_dir/Licencas/"
  file "$package_dir/OfuscadorElite.app/Contents/MacOS/OfuscadorElite" | grep -q "$target"
  if [[ "$target" == "x86_64" ]]; then
    /usr/bin/arch -x86_64 "$package_dir/OfuscadorElite.app/Contents/MacOS/OfuscadorElite" --self-test
  else
    "$package_dir/OfuscadorElite.app/Contents/MacOS/OfuscadorElite" --self-test
  fi
  /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$package_dir" "$DIST_ROOT/OfuscadorElite-macOS-$arch-v$VERSION.zip"
}

build_arch arm64 arm64
if [[ "$SKIP_INTEL" != "1" ]]; then
  build_arch x64 x86_64
else
  echo "OFUSCADOR_SKIP_INTEL=1: o pacote Intel não foi gerado."
fi
SUMS_TMP="$DIST_ROOT/.SHA256SUMS.txt.partial"
: > "$SUMS_TMP"
for artifact in \
  "$DIST_ROOT/OfuscadorElite-IA-Legendas-v1.zip" \
  "$DIST_ROOT/OfuscadorElite-Windows-x64-v$VERSION.zip" \
  "$DIST_ROOT/OfuscadorElite-macOS-arm64-v$VERSION.zip" \
  "$DIST_ROOT/OfuscadorElite-macOS-x64-v$VERSION.zip"; do
  if [[ -f "$artifact" ]]; then
    hash="$(/usr/bin/shasum -a 256 "$artifact" | /usr/bin/awk '{print toupper($1)}')"
    printf '%s  %s\n' "$hash" "$(basename "$artifact")" >> "$SUMS_TMP"
  fi
done
/bin/mv "$SUMS_TMP" "$DIST_ROOT/SHA256SUMS.txt"
echo "Pacotes criados em $DIST_ROOT"
