# Sets up the project from a fresh clone:
#   1. Installs `uv` if missing
#   2. Materializes the Python env from uv.lock
#   3. Downloads + extracts the assets archive from Google Drive (if not present)
#   4. Seeds .env from .env.example
#
# Usage (any one):
#   powershell -ExecutionPolicy Bypass -File setup.ps1     # Windows PowerShell 5.1 (built in)
#   pwsh -File setup.ps1                                   # PowerShell 7+ (if installed)
#
# Override the Google Drive file with $env:GDRIVE_FILE_ID = '<id>' if it changes.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$DefaultGDriveFileId = '1SdLBIfSK9jKPl4Z3y95bds6jrIdt0_sG'
if (-not $env:GDRIVE_FILE_ID) { $env:GDRIVE_FILE_ID = $DefaultGDriveFileId }
$ArchiveName = 'skill-aligned-eval-assets.tar.gz'
$ExpectedSha256File = 'dist/SHA256SUMS'

function Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Warn($msg)  { Write-Host "[warn] $msg" -ForegroundColor Yellow }

# 1) uv ----------------------------------------------------------------------
Step 'Checking uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step 'Installing uv'
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

# 2) Python env --------------------------------------------------------------
Step 'Syncing Python environment from uv.lock'
uv sync

# 3) Assets ------------------------------------------------------------------
Step 'Checking assets/'
$assetsEmpty = -not (Test-Path assets) -or ((Get-ChildItem assets -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)
if ($assetsEmpty) {
    if ($env:GDRIVE_FILE_ID -like 'PLACEHOLDER*') {
        Warn 'GDRIVE_FILE_ID is a placeholder; the dataset has not been published yet.'
        Warn "Set `$env:GDRIVE_FILE_ID = '<id>' or drop $ArchiveName into dist\ before re-running."
    } else {
        New-Item -ItemType Directory -Force -Path dist | Out-Null
        $archive = "dist/$ArchiveName"
        if (-not (Test-Path $archive)) {
            Step "Downloading dataset from Google Drive (id=$($env:GDRIVE_FILE_ID))"
            uv run gdown $env:GDRIVE_FILE_ID -O $archive
        }
        if (-not (Test-Path $archive)) {
            Warn "Expected $archive after download; check that the Drive file is shared with 'Anyone with the link'."
            exit 1
        }
        if (Test-Path $ExpectedSha256File) {
            Step 'Verifying sha256'
            $expected = (Get-Content $ExpectedSha256File | Select-Object -First 1).Split(' ')[0]
            $actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
            if ($expected -ne $actual) { throw "sha256 mismatch: expected $expected got $actual" }
        }
        Step 'Extracting'
        tar -xzf $archive
    }
} else {
    Write-Host 'assets/ already populated; skipping download.'
}

# 4) .env --------------------------------------------------------------------
Step 'Configuring .env'
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Warn 'Created .env from .env.example; fill in OPENAI_API_KEY before running the LLM pipeline.'
} else {
    Write-Host '.env already exists; leaving untouched.'
}

# 5) Next steps --------------------------------------------------------------
@'

Setup complete. Next steps:

  Image evaluation app   :  uv run python -m apps.image_evaluation_app   (http://localhost:5002)
  Prompt selection app   :  uv run python -m apps.prompts_selection_app
  Prompt viewer app      :  uv run python -m apps.prompts_viewer_app
  Run analysis scripts   :  uv run python analysis/<script>.py
  Re-run LLM evaluation  :  uv run python scripts/automated_llm_evaluation.py --task-id full_evaluation

'@ | Write-Host
