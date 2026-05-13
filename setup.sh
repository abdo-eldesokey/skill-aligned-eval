#!/usr/bin/env bash
# Sets up the project from a fresh clone:
#   1. Installs `uv` if missing
#   2. Materializes the Python env from uv.lock
#   3. Downloads + extracts the assets archive from Google Drive (if not present)
#   4. Seeds .env from .env.example
#
# Usage:
#   bash setup.sh
#
# Override the Google Drive file with GDRIVE_FILE_ID=... if it changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DEFAULT_GDRIVE_FILE_ID="1SdLBIfSK9jKPl4Z3y95bds6jrIdt0_sG"
GDRIVE_FILE_ID="${GDRIVE_FILE_ID:-$DEFAULT_GDRIVE_FILE_ID}"
ARCHIVE_NAME="skill-aligned-eval-assets.tar.gz"
EXPECTED_SHA256_FILE="dist/SHA256SUMS"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$1"; }

# 1) uv ----------------------------------------------------------------------
step "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
    step "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2) Python env --------------------------------------------------------------
step "Syncing Python environment from uv.lock"
uv sync

# 3) Assets ------------------------------------------------------------------
step "Checking assets/"
if [ ! -d assets ] || [ -z "$(ls -A assets 2>/dev/null)" ]; then
    if [[ "$GDRIVE_FILE_ID" == PLACEHOLDER* ]]; then
        warn "GDRIVE_FILE_ID is a placeholder; the dataset has not been published yet."
        warn "Set GDRIVE_FILE_ID=<id> or place $ARCHIVE_NAME in dist/ before re-running."
    else
        mkdir -p dist
        archive="dist/$ARCHIVE_NAME"
        if [ ! -f "$archive" ]; then
            step "Downloading dataset from Google Drive (id=$GDRIVE_FILE_ID)"
            uv run gdown "$GDRIVE_FILE_ID" -O "$archive"
        fi
        if [ ! -f "$archive" ]; then
            warn "Expected $archive after download; check that the Drive file is shared with 'Anyone with the link'."
            exit 1
        fi
        if [ -f "$EXPECTED_SHA256_FILE" ]; then
            step "Verifying sha256"
            (cd dist && sha256sum -c "$(basename "$EXPECTED_SHA256_FILE")")
        fi
        step "Extracting"
        tar -xzf "$archive"
    fi
else
    echo "assets/ already populated; skipping download."
fi

# 4) .env --------------------------------------------------------------------
step "Configuring .env"
if [ ! -f .env ]; then
    cp .env.example .env
    warn "Created .env from .env.example; fill in OPENAI_API_KEY before running the LLM pipeline."
else
    echo ".env already exists; leaving untouched."
fi

# 5) Next steps --------------------------------------------------------------
cat <<'EOF'

Setup complete. Next steps:

  Image evaluation app   :  uv run python -m apps.image_evaluation_app   (http://localhost:5002)
  Prompt selection app   :  uv run python -m apps.prompts_selection_app
  Prompt viewer app      :  uv run python -m apps.prompts_viewer_app
  Run analysis scripts   :  uv run python analysis/<script>.py
  Re-run LLM evaluation  :  uv run python scripts/automated_llm_evaluation.py --task-id full_evaluation

EOF
