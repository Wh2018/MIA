#!/usr/bin/env bash
# Build the downstream data artifacts (memory → instruction set) starting from
# the EToM + PFD JSONs. Used for both ESC and CPsy:
#
#   - ESC: legacy IJCAI shipped these artifacts. You only need this script
#          if you want to regenerate them from scratch.
#   - CPsy: legacy IJCAI never shipped them. You must run this before
#           scripts/train.sh CPsy.
#
# Inputs (must already exist):
#   data/etom/{esc,cpsy}_seven.json    EToM 7-factor JSON
#   data/pfd/{esc,cpsy}_pfd.json       PFD-annotated JSON
#
# Outputs:
#   data/memory/{esc,cpsy}/updated.json          memory after Eq.5–8
#   data/memory/{esc,cpsy}/updated_pfd.json      memory + PFD merged
#   data/instruction/{esc,cpsy}/{train,test}.json
#
# Requires an OpenAI-compatible LLM endpoint reachable at the URL set in
# config/llm_api.yaml — the ORM step (Eq.8) calls model_judge per turn to
# decide whether memory entries are outdated. GPU id comes from
# config/training.yaml.
#
# Usage:
#   bash scripts/prepare_data.sh ESC                 # full pipeline (steps 5-7)
#   bash scripts/prepare_data.sh CPsy                # full pipeline (steps 5-7)
#   STEP=memory bash scripts/prepare_data.sh CPsy    # only memory update
#   STEP=merge  bash scripts/prepare_data.sh CPsy    # only PFD merge
#   STEP=inst   bash scripts/prepare_data.sh CPsy    # only instruction build
#
# Ablation flags:
#   NO_ORM=1 bash scripts/prepare_data.sh CPsy       # skip Eq.8 (faster)
#   NO_SIM=1 bash scripts/prepare_data.sh CPsy       # skip Eq.7

set -euo pipefail

DATASET="${1:-}"
if [ -z "${DATASET}" ]; then
    echo "Usage: $0 <ESC|CPsy>" >&2
    exit 1
fi

case "${DATASET}" in
    ESC|esc)   DATASET_LC="esc" ;;
    CPsy|cpsy) DATASET_LC="cpsy" ;;
    *) echo "Unknown dataset: ${DATASET} (use ESC or CPsy)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$(python scripts/_get_config.py training.yaml cuda_device --root "${PROJECT_ROOT}")"

STEP="${STEP:-all}"
SEVEN_JSON="data/etom/${DATASET_LC}_seven.json"
PFD_JSON="data/pfd/${DATASET_LC}_pfd.json"
MEM_DIR="data/memory/${DATASET_LC}"
INST_DIR="data/instruction/${DATASET_LC}"

if [ ! -f "${SEVEN_JSON}" ]; then
    echo "ERROR: ${SEVEN_JSON} not found. Run EToM (steps 1-2) first." >&2
    exit 1
fi

mkdir -p "${MEM_DIR}" "${INST_DIR}"

EXTRA_FLAGS=()
[ "${NO_SIM:-0}" = "1" ] && EXTRA_FLAGS+=("--no-similarity")
[ "${NO_ORM:-0}" = "1" ] && EXTRA_FLAGS+=("--no-orm")

step_memory() {
    echo "=== [5/7] Memory update (Eq.5–8) → ${MEM_DIR}/updated.json ==="
    python -m src.memory.run_update \
        --input "${SEVEN_JSON}" \
        --output "${MEM_DIR}/updated.json" \
        --similar-log  "${MEM_DIR}/similar_log.json" \
        --outdated-log "${MEM_DIR}/outdated_log.json" \
        --sim-score-log "${MEM_DIR}/sim_score_log.json" \
        "${EXTRA_FLAGS[@]}"
}

step_merge() {
    echo "=== [6/7] Merge PFD into memory → ${MEM_DIR}/updated_pfd.json ==="
    if [ ! -f "${MEM_DIR}/updated.json" ]; then
        echo "ERROR: ${MEM_DIR}/updated.json missing. Run STEP=memory first." >&2
        exit 1
    fi
    if [ ! -f "${PFD_JSON}" ]; then
        echo "ERROR: ${PFD_JSON} not found." >&2
        exit 1
    fi
    python -m src.instruction.add_pfd \
        --memory "${MEM_DIR}/updated.json" \
        --pfd "${PFD_JSON}" \
        --output "${MEM_DIR}/updated_pfd.json"
}

step_inst() {
    echo "=== [7/7] Instruction set → ${INST_DIR}/{train,test}.json ==="
    if [ ! -f "${MEM_DIR}/updated_pfd.json" ]; then
        echo "ERROR: ${MEM_DIR}/updated_pfd.json missing. Run STEP=merge first." >&2
        exit 1
    fi
    python -m src.instruction.upd_pfd2inst \
        --input "${MEM_DIR}/updated_pfd.json" \
        --train-out "${INST_DIR}/train.json" \
        --test-out  "${INST_DIR}/test.json"
}

case "${STEP}" in
    memory) step_memory ;;
    merge)  step_merge ;;
    inst)   step_inst ;;
    all)    step_memory; step_merge; step_inst ;;
    *) echo "Unknown STEP=${STEP} (use memory|merge|inst|all)" >&2; exit 1 ;;
esac

echo "Done. Next: bash scripts/train.sh ${DATASET}"
