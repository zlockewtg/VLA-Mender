#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly RUN_EVAL="${SCRIPT_DIR}/run_eval.sh"
readonly PROFILE="${SCRIPT_DIR}/configs/stove_pan_scene3_official50.yaml"
readonly OPENPI_ROOT="/mnt/public/tgy/VLA-Mender/third_party/openpi"
readonly OPENPI_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"
readonly OPENPI_PYTHON="/opt/venv/openpi/bin/python"
readonly CHECKPOINT_BASE="${VLA_MENDER_EVAL_CHECKPOINT_BASE:-/mnt/public/tgy/VLA-Mender/checkpoints/pi0_libero_stove_pan_scene3_quality40/prefix_repair_20hz_h50_lr2e5_4k_v1}"
readonly OUTPUT_BASE="${VLA_MENDER_EVAL_OUTPUT_BASE:-/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair_v2_contact_path_review/eval/checkpoints}"
readonly CONFIG_NAME="pi0_libero"
readonly TRAINING_SETTINGS="${VLA_MENDER_EVAL_TRAINING_SETTINGS:-/mnt/public/tgy/VLA-Mender/outputs/stove_pan_scene3_repair_v2_contact_path_review/dataset_quality40_v1/training.resolved.yaml}"
readonly REPO_ID_PREFIX="${VLA_MENDER_EVAL_REPO_ID_PREFIX:-local/stove-pan-scene3}"
MAX_STEPS="$(awk '$1 == "max_steps:" { print $2; exit }' "${PROFILE}")"
readonly MAX_STEPS
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid or missing evaluation.max_steps in ${PROFILE}" >&2
    exit 2
}

wait_for_training() {
    local pattern
    pattern="^/opt/venv/openpi/bin/python3 -u -m workflow.training.openpi_runner --settings ${TRAINING_SETTINGS}$"
    while pgrep -f "${pattern}" >/dev/null; do
        echo "[$(date -u +%FT%TZ)] training still active; waiting before checkpoint evaluation"
        sleep 30
    done
}

discover_steps() {
    find "${CHECKPOINT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^[0-9]+$/' \
        | sort -n
}

run_step() {
    local step="$1"
    local checkpoint="${CHECKPOINT_BASE}/${step}"
    local output="${OUTPUT_BASE}/step_${step}_official50_seed7_20hz_max${MAX_STEPS}"
    local args=()

    [[ -f "${checkpoint}/model.safetensors" ]] || {
        echo "Missing checkpoint weights: ${checkpoint}" >&2
        return 2
    }
    [[ -f "${checkpoint}/assets/physical-intelligence/libero/norm_stats.json" ]] || {
        echo "Missing checkpoint norm stats: ${checkpoint}" >&2
        return 2
    }
    if [[ -f "${output}/summary.json" ]] && jq -e \
        '((.failed_tasks // []) | length == 0) and
         ((.episodes // 0) >= (.requested_episodes // 1))' \
        "${output}/summary.json" >/dev/null; then
        echo "[$(date -u +%FT%TZ)] skipping completed step ${step}: ${output}"
        return 0
    fi
    if [[ -d "${output}" ]]; then
        args+=(--resume)
    fi
    echo "[$(date -u +%FT%TZ)] evaluating step ${step}"
    (
        cd "${REPO_ROOT}"
        env OPENPI_PYTHON="${OPENPI_PYTHON}" "${RUN_EVAL}" "${PROFILE}" \
            --checkpoint "${checkpoint}" \
            --config-name "${CONFIG_NAME}" \
            --openpi-source "${OPENPI_ROOT}" \
            --openpi-commit "${OPENPI_COMMIT}" \
            --python "${OPENPI_PYTHON}" \
            --output "${output}" \
            --repo-id-prefix "${REPO_ID_PREFIX}-step${step}" \
            "${args[@]}"
    )
    echo "[$(date -u +%FT%TZ)] completed step ${step}: ${output}"
}

if [[ "${WAIT_FOR_TRAINING:-0}" == "1" ]]; then
    wait_for_training
fi

if [[ "$#" -gt 0 ]]; then
    steps=("$@")
else
    mapfile -t steps < <(discover_steps)
fi
[[ "${#steps[@]}" -gt 0 ]] || { echo "No checkpoints found in ${CHECKPOINT_BASE}" >&2; exit 2; }

for step in "${steps[@]}"; do
    [[ "${step}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint step: ${step}" >&2; exit 2; }
    run_step "${step}"
done
