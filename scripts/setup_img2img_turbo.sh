#!/bin/bash
# Install the project environment and pin the official img2img-turbo checkout.

set -euo pipefail
trap 'echo "ERROR: setup failed at line $LINENO: $BASH_COMMAND" >&2' ERR

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
[ -f "$COMPAT_PATCH" ] || {
    echo "ERROR: Compatibility patch not found: $COMPAT_PATCH" >&2
    echo "Update the project checkout before running setup again." >&2
    exit 1
}
"$PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else "Python 3.10 is required; found " + sys.version.split()[0])'

mkdir -p "$PROJECT_ROOT/external"
if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
    git clone "$UPSTREAM_URL" "$EXTERNAL_ROOT"
fi

if ! git -C "$EXTERNAL_ROOT" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
    echo "Fetching pinned img2img-turbo commit..."
    git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$UPSTREAM_COMMIT"
fi

CURRENT_COMMIT="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
if [ "$CURRENT_COMMIT" != "$UPSTREAM_COMMIT" ]; then
    if [ -n "$(git -C "$EXTERNAL_ROOT" status --porcelain)" ]; then
        echo "ERROR: img2img-turbo has unexpected local changes at $CURRENT_COMMIT." >&2
        echo "Remove external/img2img-turbo and rerun setup." >&2
        exit 1
    fi
    git -C "$EXTERNAL_ROOT" checkout --detach "$UPSTREAM_COMMIT"
fi
echo "Pinned img2img-turbo to $UPSTREAM_COMMIT"

if git -C "$EXTERNAL_ROOT" apply --reverse --check "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "Compatibility patch already applied."
elif git -C "$EXTERNAL_ROOT" apply --check "$COMPAT_PATCH" >/dev/null 2>&1; then
    git -C "$EXTERNAL_ROOT" apply --check "$COMPAT_PATCH"
    git -C "$EXTERNAL_ROOT" apply "$COMPAT_PATCH"
    echo "Applied pix2pix FP16 master-weight compatibility patch."
else
    echo "ERROR: The compatibility patch is neither applicable nor already applied." >&2
    echo "Current external changes:" >&2
    git -C "$EXTERNAL_ROOT" status --short >&2
    echo "Remove external/img2img-turbo and rerun setup." >&2
    exit 1
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
