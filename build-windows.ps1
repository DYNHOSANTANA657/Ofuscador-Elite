param(
    [string]$PythonPath = "",
    [string]$FfmpegDirectory = "",
    [string]$PnpmPath = "",
    [string]$SubtitleModelUrl = "",
    [string]$SubtitleModelSha256 = "",
    [switch]$UseCompiledInterface,
    [switch]$UseExistingEnvironment
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot ".build-windows"
$FinalDirectory = Join-Path $ProjectRoot "dist"
$FinalZip = Join-Path $FinalDirectory "OfuscadorElite-Windows-x64-v1.3.0.zip"

if (Test-Path -LiteralPath $BuildRoot) { throw "A pasta temporária $BuildRoot já existe. Renomeie ou remova essa pasta antes de reconstruir." }
if (Test-Path -LiteralPath $FinalZip) { throw "O arquivo $FinalZip já existe. Ele não será substituído." }
New-Item -ItemType Directory -Path $BuildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $FinalDirectory | Out-Null
if (($SubtitleModelUrl -and -not $SubtitleModelSha256) -or ($SubtitleModelSha256 -and -not $SubtitleModelUrl)) { throw "Informe juntos SubtitleModelUrl e SubtitleModelSha256." }
if ($SubtitleModelSha256 -and $SubtitleModelSha256 -notmatch '^[A-Fa-f0-9]{64}$') { throw "SubtitleModelSha256 precisa conter 64 caracteres hexadecimais." }
$PublicationConfig = Join-Path $BuildRoot "publication.json"
$PublicationJson = @{ subtitleModelUrl = $SubtitleModelUrl; subtitleModelSha256 = $SubtitleModelSha256.ToLowerInvariant() } | ConvertTo-Json
[System.IO.File]::WriteAllText($PublicationConfig, $PublicationJson, [System.Text.UTF8Encoding]::new($false))

if (-not $UseCompiledInterface -and -not $PnpmPath) {
    $PnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
    if ($PnpmCommand) { $PnpmPath = $PnpmCommand.Source }
}
if ($PnpmPath) {
    $PnpmPath = (Resolve-Path -LiteralPath $PnpmPath).Path
    & $PnpmPath run build:portable
    if ($LASTEXITCODE -ne 0) { throw "A interface não pôde ser compilada." }
    & $PnpmPath test
    if ($LASTEXITCODE -ne 0) { throw "Os testes da interface falharam." }
    & $PnpmPath run typecheck:portable
    if ($LASTEXITCODE -ne 0) { throw "A verificação TypeScript da interface falhou." }
} elseif (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "backend\static\index.html"))) {
    throw "Instale Node.js e pnpm para compilar a interface."
}

if (-not $PythonPath) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) { throw "Python 3.12 x64 não foi encontrado. Use -PythonPath para informar o executável." }
    $PythonPath = $PythonCommand.Source
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
& $PythonPath -c "import platform,sys; assert sys.version_info >= (3,11); assert platform.architecture()[0] == '64bit'"

if ($UseExistingEnvironment) {
    $BuildPython = $PythonPath
    & $BuildPython -c "import PyInstaller,fastapi,keyring,piper,onnxruntime,cv2,rapidocr"
} else {
    $Venv = Join-Path $BuildRoot "venv"
    & $PythonPath -m venv $Venv
    $BuildPython = Join-Path $Venv "Scripts\python.exe"
    & $BuildPython -m pip install --disable-pip-version-check -r (Join-Path $ProjectRoot "backend\requirements-build.txt")
}

if ($FfmpegDirectory) {
    $Ffmpeg = Join-Path $FfmpegDirectory "ffmpeg.exe"
    $Ffprobe = Join-Path $FfmpegDirectory "ffprobe.exe"
} else {
    $Ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
    $Ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
}
$Ffmpeg = (Resolve-Path -LiteralPath $Ffmpeg).Path
$Ffprobe = (Resolve-Path -LiteralPath $Ffprobe).Path
if (-not (& $Ffmpeg -hide_banner -encoders 2>&1 | Select-String -SimpleMatch "libx264")) { throw "O FFmpeg informado não contém o codificador H.264 libx264." }
$SubtitleModelPackage = Join-Path $FinalDirectory "OfuscadorElite-IA-Legendas-v1.zip"
if (-not (Test-Path -LiteralPath $SubtitleModelPackage)) { throw "Gere primeiro OfuscadorElite-IA-Legendas-v1.zip com scripts\build-subtitle-model-pack.py." }
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
try {
    & $BuildPython (Join-Path $ProjectRoot "scripts\verify-subtitle-model-pack.py") $SubtitleModelPackage
    if ($LASTEXITCODE -ne 0) { throw "O diagnóstico de RapidOCR e LaMa falhou." }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

$PreviousPath = $env:PATH
$env:PATH = "$(Split-Path -Parent $Ffmpeg);$PreviousPath"
try {
    Push-Location (Join-Path $ProjectRoot "backend")
    try {
        & $BuildPython -m pytest "tests" -q
        if ($LASTEXITCODE -ne 0) { throw "Os testes do backend falharam." }
    } finally {
        Pop-Location
    }
} finally {
    $env:PATH = $PreviousPath
}

$PyInstallerDist = Join-Path $BuildRoot "pyinstaller-dist"
$PyInstallerWork = Join-Path $BuildRoot "pyinstaller-work"
$SpecDirectory = Join-Path $BuildRoot "spec"
$Separator = ";"
& $BuildPython -m PyInstaller --noconfirm --clean --windowed --onedir --name OfuscadorElite `
    --paths (Join-Path $ProjectRoot "backend") `
    --distpath $PyInstallerDist --workpath $PyInstallerWork --specpath $SpecDirectory `
    --add-data "$(Join-Path $ProjectRoot 'backend\static')${Separator}static" `
    --add-data "$(Join-Path $ProjectRoot 'backend\voices')${Separator}voices" `
    --add-data "${PublicationConfig}${Separator}." `
    --add-binary "${Ffmpeg}${Separator}bin" --add-binary "${Ffprobe}${Separator}bin" `
    --hidden-import keyring.backends.Windows --collect-all keyring --collect-all piper --collect-all onnxruntime --collect-all rapidocr --collect-all cv2 `
    (Join-Path $ProjectRoot "backend\launcher_entry.py")
if ($LASTEXITCODE -ne 0) { throw "O executável do Windows não pôde ser criado." }

$Package = Join-Path $BuildRoot "OfuscadorElite-Windows-x64"
New-Item -ItemType Directory -Path $Package | Out-Null
Move-Item -LiteralPath (Join-Path $PyInstallerDist "OfuscadorElite") -Destination (Join-Path $Package "OfuscadorElite")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "LEIA-ME.txt") -Destination $Package
Copy-Item -LiteralPath (Join-Path $ProjectRoot "PUBLICACAO-HOSTINGER.txt") -Destination $Package
New-Item -ItemType Directory -Path (Join-Path $Package "Licencas") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "licenses\FFmpeg-GPLv3.txt") -Destination (Join-Path $Package "Licencas\FFmpeg-GPLv3.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "licenses\Piper-GPLv3.txt") -Destination (Join-Path $Package "Licencas\Piper-GPLv3.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "backend\voices\MODEL_CARD-pt_BR-faber-medium.txt") -Destination (Join-Path $Package "Licencas\Modelo-Faber-CC0.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "licenses\AVISOS-DE-TERCEIROS.txt") -Destination (Join-Path $Package "Licencas\AVISOS-DE-TERCEIROS.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "licenses\IA-Legendas-APACHE-MIT.txt") -Destination (Join-Path $Package "Licencas\IA-Legendas-APACHE-MIT.txt")

$Executable = Join-Path $Package "OfuscadorElite\OfuscadorElite.exe"
$SelfTestProcess = Start-Process -FilePath $Executable -ArgumentList "--self-test" -PassThru -Wait -WindowStyle Hidden
if ($SelfTestProcess.ExitCode -ne 0) { throw "O teste interno do pacote Windows falhou." }
[System.Reflection.Assembly]::LoadWithPartialName("System.IO.Compression.FileSystem") | Out-Null
$ZipCreated = $false
$LastZipError = $null
for ($Attempt = 1; $Attempt -le 5 -and -not $ZipCreated; $Attempt++) {
    if (Test-Path -LiteralPath $FinalZip) { Remove-Item -LiteralPath $FinalZip -Force }
    Start-Sleep -Milliseconds (500 * $Attempt)
    try {
        [System.IO.Compression.ZipFile]::CreateFromDirectory($Package, $FinalZip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
        $Archive = [System.IO.Compression.ZipFile]::OpenRead($FinalZip)
        try {
            $ExpectedEntry = $Archive.Entries | Where-Object { ($_.FullName -replace '\\','/') -eq 'OfuscadorElite-Windows-x64/OfuscadorElite/OfuscadorElite.exe' }
            if (-not $ExpectedEntry -or $ExpectedEntry.Length -le 0) { throw "O ZIP criado não contém o executável esperado." }
        } finally {
            $Archive.Dispose()
        }
        $ZipCreated = $true
    } catch {
        $LastZipError = $_
    }
}
if (-not $ZipCreated) { throw "O pacote Windows não pôde ser compactado após cinco tentativas: $LastZipError" }
$WindowsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $FinalZip).Hash
$SumsFile = Join-Path $FinalDirectory "SHA256SUMS.txt"
$SumsLines = @()
if (Test-Path -LiteralPath $SumsFile) { $SumsLines = @(Get-Content -LiteralPath $SumsFile | Where-Object { $_ -notmatch "  $([regex]::Escape((Split-Path -Leaf $FinalZip)))$" }) }
$SumsLines += "$WindowsHash  $(Split-Path -Leaf $FinalZip)"
[System.IO.File]::WriteAllText($SumsFile, (($SumsLines -join "`n") + "`n"), [System.Text.UTF8Encoding]::new($false))
Write-Host "Pacote criado: $FinalZip"
Write-Host "SHA-256: $WindowsHash"
