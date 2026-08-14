#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG_DIR="${SCRIPT_DIR}/configs"
readonly PYTHON_BIN="${OPENPI_PYTHON:-/opt/venv/openpi/bin/python}"

config_arg="${1:-default}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

if [[ "${config_arg}" == *.yaml || "${config_arg}" == *.yml || "${config_arg}" == */* ]]; then
    config_path="${config_arg}"
else
    config_path="${CONFIG_DIR}/${config_arg}.yaml"
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    echo "Available profiles:" >&2
    find "${CONFIG_DIR}" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -printf '  %f\n' >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "OpenPI Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" --config "${config_path}" "$@"
