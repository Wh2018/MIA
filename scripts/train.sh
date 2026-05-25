#!/usr/bin/env bash
# LoRA SFT + prediction for the MIA datasets.
#
# Paths and GPU id are read from config/training.yaml — edit that file once
# for your machine. Per-dataset outputs always land under
#   data/training/<dataset_lowercase>/sft/
#   data/training/<dataset_lowercase>/predict/
# so an ESC and CPsy run cannot collide.
#
# Usage:
#   bash scripts/train.sh ESC                # SFT then predict
#   bash scripts/train.sh CPsy               # SFT then predict
#   STAGE=sft     bash scripts/train.sh ESC  # SFT only
#   STAGE=predict bash scripts/train.sh ESC  # predict only (needs trained adapter)

set -euo pipefail

DATASET="${1:-}"
if [ -z "${DATASET}" ]; then
    echo "Usage: $0 <ESC|CPsy>" >&2
    exit 1
fi

case "${DATASET}" in
    ESC|esc)   DATASET="ESC";  DATASET_LC="esc" ;;
    CPsy|cpsy) DATASET="CPsy"; DATASET_LC="cpsy" ;;
    *) echo "Unknown dataset: ${DATASET} (use ESC or CPsy)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

get_cfg() {
    python scripts/_get_config.py training.yaml "$1" --root "${PROJECT_ROOT}" "${@:2}"
}

MODEL_PATH="$(get_cfg base_model_path)"
LLAMA_FACTORY_RAW="$(get_cfg llamafactory_dir)"
export CUDA_VISIBLE_DEVICES="$(get_cfg cuda_device)"

if [ -z "${LLAMA_FACTORY_RAW}" ]; then
    echo "ERROR: llamafactory_dir is empty in config/training.yaml." >&2
    echo "Set it to your local LLaMA-Factory checkout." >&2
    exit 1
fi

case "${LLAMA_FACTORY_RAW}" in
    /*) LLAMA_FACTORY_DIR="${LLAMA_FACTORY_RAW}" ;;
    *)  LLAMA_FACTORY_DIR="${PROJECT_ROOT}/${LLAMA_FACTORY_RAW}" ;;
esac

# `base_model_path` may be either a local path or a Hugging Face model id.
# Resolve local paths when they exist; otherwise pass the value through.
if [ -d "${PROJECT_ROOT}/${MODEL_PATH}" ]; then
    MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
fi

# Per-dataset output policy — fixed, not configurable, so ESC and CPsy can
# never overwrite each other.
DATASET_DIR="${PROJECT_ROOT}/data/instruction"
ADAPTER_DIR="${PROJECT_ROOT}/data/training/${DATASET_LC}/sft"
PREDICT_DIR="${PROJECT_ROOT}/data/training/${DATASET_LC}/predict"
STAGE="${STAGE:-all}"

if [ ! -d "${LLAMA_FACTORY_DIR}" ]; then
    echo "ERROR: llamafactory_dir='${LLAMA_FACTORY_DIR}' (from config/training.yaml) does not exist." >&2
    echo "Clone from https://github.com/hiyouga/LLaMA-Factory and edit config/training.yaml." >&2
    exit 1
fi
if [ ! -f "${DATASET_DIR}/${DATASET_LC}/train.json" ]; then
    echo "ERROR: missing ${DATASET_DIR}/${DATASET_LC}/train.json." >&2
    echo "Run scripts/prepare_data.sh ${DATASET} first." >&2
    exit 1
fi

python scripts/_register_datasets.py --root "${PROJECT_ROOT}" --dataset-dir data/instruction

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "ERROR: llamafactory-cli not on PATH. Install with:" >&2
    echo "  cd ${LLAMA_FACTORY_DIR} && pip install -e '.[torch,metrics]'" >&2
    exit 1
fi

run_sft() {
    echo "=== SFT: ${DATASET}_train → ${ADAPTER_DIR} ==="
    mkdir -p "${ADAPTER_DIR}"
    llamafactory-cli train "${PROJECT_ROOT}/training/qwen25_lora_sft.yaml" \
        model_name_or_path="${MODEL_PATH}" \
        dataset_dir="${DATASET_DIR}" \
        dataset="${DATASET}_train" \
        output_dir="${ADAPTER_DIR}"
}

run_predict() {
    if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
        echo "ERROR: no adapter at ${ADAPTER_DIR}. Run STAGE=sft first." >&2
        exit 1
    fi
    echo "=== Predict: ${DATASET}_test → ${PREDICT_DIR} ==="
    mkdir -p "${PREDICT_DIR}"
    llamafactory-cli train "${PROJECT_ROOT}/training/qwen25_lora_predict.yaml" \
        model_name_or_path="${MODEL_PATH}" \
        dataset_dir="${DATASET_DIR}" \
        eval_dataset="${DATASET}_test" \
        adapter_name_or_path="${ADAPTER_DIR}" \
        output_dir="${PREDICT_DIR}"
}

case "${STAGE}" in
    sft)     run_sft ;;
    predict) run_predict ;;
    all)     run_sft; run_predict ;;
    *) echo "Unknown STAGE=${STAGE} (use sft|predict|all)" >&2; exit 1 ;;
esac

echo "Done. Adapter: ${ADAPTER_DIR}  |  Predictions: ${PREDICT_DIR}"
