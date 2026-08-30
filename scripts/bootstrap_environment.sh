#!/usr/bin/env bash
# Reproduce the CaP-X-style VLA-Mender environment from a fresh checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VLA_MENDER_VENV:-$ROOT/.venv-libero}"
START_SERVERS=1
EXTRA=libero

while (($#)); do
  case "$1" in
    --no-server-smoke) START_SERVERS=0 ;;
    --extra) shift; EXTRA="${1:?missing value for --extra}" ;;
    --venv) shift; VENV_PATH="${1:?missing value for --venv}" ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/bootstrap_environment.sh [options]

Options:
  --venv PATH          virtual environment (default: .venv-libero)
  --extra NAME         uv extra to install (default: libero)
  --no-server-smoke    install and import-check only; do not start tools

The script initializes Git submodules, installs their editable uv sources,
provisions tool checkpoints at repository-relative paths, verifies imports,
and by default starts the three local tool services in the background through
scripts/verify_tools.py.
EOF
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v git >/dev/null || { echo 'ERROR: git is required' >&2; exit 127; }
command -v uv >/dev/null || { echo 'ERROR: uv is required' >&2; exit 127; }

cd "$ROOT"
git submodule sync --recursive
git submodule update --init --recursive

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  uv venv "$VENV_PATH" --python 3.12
fi
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --active --locked --extra "$EXTRA"
python scripts/install_tool_checkpoints.py
python scripts/verify_environment.py --libero

if (( START_SERVERS )); then
  python scripts/verify_tools.py --server-smoke
fi
