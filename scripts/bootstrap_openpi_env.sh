#!/usr/bin/env bash
# Build the isolated OpenPI runtime documented by third_party/openpi/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_SOURCE="${VLA_MENDER_OPENPI_SOURCE:-$ROOT/third_party/openpi}"
OPENPI_ENV="${VLA_MENDER_OPENPI_ENV:-$ROOT/.venv-openpi}"
UV_CACHE="${VLA_MENDER_UV_CACHE:-$ROOT/.uv-cache}"
PIN_FILE="$ROOT/third_party/openpi.commit"

command -v uv >/dev/null || { echo "uv is required; see OpenPI README" >&2; exit 2; }
test -f "$OPENPI_SOURCE/pyproject.toml" || { echo "invalid OpenPI checkout: $OPENPI_SOURCE" >&2; exit 2; }
test -f "$PIN_FILE" || { echo "missing OpenPI pin: $PIN_FILE" >&2; exit 2; }
actual="$(git -C "$OPENPI_SOURCE" rev-parse HEAD)"
expected="$(tr -d '[:space:]' < "$PIN_FILE")"
[[ "$actual" == "$expected" ]] || { echo "OpenPI checkout $actual != pinned $expected" >&2; exit 3; }

# The official workflow uses GIT_LFS_SKIP_SMUDGE=1, uv sync, and an editable
# install.  Keep its environment separate from the LIBERO/VLA-Mender env.
GIT_LFS_SKIP_SMUDGE=1 UV_CACHE_DIR="$UV_CACHE" UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$OPENPI_ENV" uv sync --frozen --project "$OPENPI_SOURCE" --python 3.11
GIT_LFS_SKIP_SMUDGE=1 UV_CACHE_DIR="$UV_CACHE" uv pip install --python "$OPENPI_ENV/bin/python" -e "$OPENPI_SOURCE"

# The rollout is an in-process OpenPI + LIBERO worker. Install the simulator
# bridge into this same Python 3.11 environment; do not mix it with .venv-libero2.
UV_CACHE_DIR="$UV_CACHE" uv pip install --python "$OPENPI_ENV/bin/python" \
  -e "$ROOT/third_party/LIBERO-PRO" \
  -e "$ROOT/third_party/libero_dependencies/robosuite" \
  "mujoco==3.8.1" "gym==0.25.2" "easydict==1.9" "robomimic==0.2.0" \
  "bddl==1.0.1" "future==0.18.2" "cloudpickle==2.1.0" \
  "thop==0.1.1-2209072238" "pyyaml" "imageio[ffmpeg]"

# PyTorch inference requires the source patches called out in the official
# README.  Copy into this environment only (never into uv's shared cache).
site="$($OPENPI_ENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "$OPENPI_SOURCE/src/openpi/models_pytorch/transformers_replace" ]]; then
  cp -r "$OPENPI_SOURCE/src/openpi/models_pytorch/transformers_replace/." "$site/transformers/"
fi

VLA_MENDER_OPENPI_ENV="$OPENPI_ENV" "$OPENPI_ENV/bin/python" "$ROOT/scripts/check_openpi_runtime.py"   --source "$OPENPI_SOURCE" --json-out "$ROOT/openpi-runtime-manifest.json"
echo "OpenPI runtime ready: $OPENPI_ENV"
