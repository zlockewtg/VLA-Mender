#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly REFERENCE_OPENPI_ROOT="/mnt/public/tgy/capx-aspire/aspire/vla_mender/openpi"
readonly OPENPI_ROOT="${OPENPI_ROOT:-${REFERENCE_OPENPI_ROOT}}"
readonly OPENPI_ENV="${VLA_MENDER_OPENPI_ENV:-/opt/venv/openpi}"
readonly PYTHON="${OPENPI_ENV}/bin/python"
readonly RUN_EVAL="${SCRIPT_DIR}/run_eval.sh"
readonly CONFIG_NAME="pi0_libero_goal_task0_repair46_align70_native_crossboundary_20hz_lr2p5e5_4k_h50_v1"
readonly EXP_NAME="goal0_r46_a70_lr2p5e5_v1"
readonly RUN_TAG="goal0_r46_a70_lr2p5e5_v1eval"
readonly CHECKPOINT_BASE="${CHECKPOINT_BASE:-${OPENPI_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}}"
readonly OUTPUT_BASE="${EVAL_OUTPUT_BASE:-/mnt/public/tgy/data}"

if [[ "$#" -eq 0 ]]; then
    set -- 500 1000 1500 2000 2500 3000 3500 4000
fi

run_step() {
    local step="$1"
    local checkpoint="${CHECKPOINT_BASE}/${step}"
    local output_root="${OUTPUT_BASE}/${RUN_TAG}_step${step}_official20_binarygripper_20hz_max500_seed7"

    [[ -x "${PYTHON}" ]] || { echo "Missing OpenPI interpreter: ${PYTHON}" >&2; return 2; }
    [[ -f "${checkpoint}/model.safetensors" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 2; }
    if [[ -f "${output_root}/summary.json" ]]; then
        echo "[$(date -u +%FT%TZ)] skipping completed step ${step}: ${output_root}"
        return 0
    fi
    [[ ! -e "${output_root}" ]] || { echo "Refusing incomplete existing output: ${output_root}" >&2; return 2; }
    echo "[$(date -u +%FT%TZ)] starting repair46_align70 step ${step}"
    (
        cd "${REPO_ROOT}"
        OPENPI_PYTHON="${PYTHON}" "${RUN_EVAL}" repair46_align70_official20 \
            --checkpoint "${checkpoint}" \
            --config-name "${CONFIG_NAME}" \
            --openpi-source "${OPENPI_ROOT}" \
            --python "${PYTHON}" \
            --output "${output_root}" \
            --repo-id-prefix "local/goal0-repair46-align70-step${step}"
    )
    echo "[$(date -u +%FT%TZ)] completed repair46_align70 step ${step}: ${output_root}"
}

for step in "$@"; do
    [[ "${step}" =~ ^[0-9]+$ ]] || { echo "Invalid step: ${step}" >&2; exit 2; }
    run_step "${step}"
done
