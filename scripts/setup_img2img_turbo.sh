#!/bin/bash
# Install the project environment and pin the official img2img-turbo checkout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
UPSTREAM_URL="https://github.com/GaParmar/img2img-turbo.git"
UPSTREAM_COMMIT="86f54146590ffb4543c8cf85b5a36657da670924"
EXTERNAL_ROOT="$PROJECT_ROOT/external/img2img-turbo"
COMPAT_PATCH="$PROJECT_ROOT/patches/img2img-turbo-pix2pix-fp16.patch"

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required." >&2
    exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
}
"$PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else "Python 3.10 is required; found " + sys.version.split()[0])'

mkdir -p "$PROJECT_ROOT/external"
if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
    git clone "$UPSTREAM_URL" "$EXTERNAL_ROOT"
fi

git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$EXTERNAL_ROOT" checkout --detach "$UPSTREAM_COMMIT"
echo "Pinned img2img-turbo to $UPSTREAM_COMMIT"

if git -C "$EXTERNAL_ROOT" apply --reverse --check "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "Compatibility patch already applied."
else
    git -C "$EXTERNAL_ROOT" apply --check "$COMPAT_PATCH"
    git -C "$EXTERNAL_ROOT" apply "$COMPAT_PATCH"
    echo "Applied pix2pix FP16 master-weight compatibility patch."
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "Skipping dependency installation because SKIP_INSTALL=1."
fi

if [ -d "$PROJECT_ROOT/data/processed/day2night" ]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_img2img_turbo_data.py"
else
    echo "Processed data not found; run scripts/download_data.sh before training."
fi

echo ""
echo "Setup complete. Validate the full launch matrix without training:"
echo "  DRY_RUN=1 bash scripts/run_all_experiments.sh"
