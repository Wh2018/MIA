#!/usr/bin/env bash
# End-to-end MIA data pipeline.
#
# Reproduces the steps used in the paper:
#   1. EToM 7-factor annotation (LLM judge + H-RAG few-shot)
#   2. Parse per-conversation .txt → single 7-factor JSON
#   3. PFD (Personal-Factual Discriminator) annotation
#   4. Parse PFD .txt → merged JSON
#   5. Memory update (Eq.5–Eq.8: append / overwrite / similarity / ORM)
#   6. Merge PFD labels into updated-memory JSON
#   7. Convert to LLaMA-Factory instruction format (9:1 split)
#
# The two datasets (ESConv / CPsyCounD) share the same shape but use slightly
# different prompts and exemplars; this script runs them sequentially. Either
# can be skipped by setting RUN_ESC=0 or RUN_CPSY=0 in the environment.
#
# Usage:
#   bash scripts/run_pipeline.sh                  # both pipelines
#   RUN_CPSY=0 bash scripts/run_pipeline.sh       # ESC only
#   ESC_INPUT=path/to/esc.json bash scripts/run_pipeline.sh
#
# Assumes the project root is the parent of this scripts/ dir, and that a
# local OpenAI-compatible server is reachable at the URL set in
# config/llm_api.yaml.

set -euo pipefail

# Resolve project root from the script's location so the script can be run
# from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Make `python -m src.*` resolve to code_new/src.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# GPU id comes from config/training.yaml (cuda_device), keeping all path /
# device settings in one place.
export CUDA_VISIBLE_DEVICES="$(python scripts/_get_config.py training.yaml cuda_device --root "${PROJECT_ROOT}")"

PY="${PY:-python}"
RUN_ESC="${RUN_ESC:-1}"
RUN_CPSY="${RUN_CPSY:-1}"

ESC_INPUT="${ESC_INPUT:-data/raw/esc/esc.json}"
CPSY_INPUT="${CPSY_INPUT:-data/raw/cpsy/cpsy.json}"

log() { printf '\n=== %s ===\n' "$*"; }

run_esc() {
    log "ESC pipeline"

    log "[1/7] EToM 7-factor generation (ESC)"
    "${PY}" -m src.etom.gen_seven_esc \
        --input "${ESC_INPUT}" \
        --output-dir data/etom/esc_raw

    log "[2/7] Parse EToM .txt → JSON (ESC)"
    "${PY}" -m src.etom.parse_seven \
        --input-dir data/etom/esc_raw \
        --output-file data/etom/esc_seven.json

    log "[3/7] PFD generation (ESC)"
    # Reuses the EToM raw dir as the mind-info source. The 4-factor portion
    # (Belief / Intention / Desire / Emotion) of each Qwen{N}.txt is what
    # `_read_mind_lines` consumes.
    "${PY}" -m src.pfd.gen_pfd_esc \
        --input "${ESC_INPUT}" \
        --mind-dir data/etom/esc_raw \
        --output-dir data/pfd/esc_raw

    log "[4/7] Parse PFD .txt → JSON (ESC)"
    "${PY}" -m src.pfd.parse_pfd \
        --style esc \
        --input-dir data/pfd/esc_raw \
        --output-file data/pfd/esc_pfd.json

    log "[5/7] Memory update with ORM (ESC)"
    mkdir -p data/memory/esc
    "${PY}" -m src.memory.run_update \
        --input data/etom/esc_seven.json \
        --output data/memory/esc/updated.json \
        --similar-log data/memory/esc/similar_log.json \
        --outdated-log data/memory/esc/outdated_log.json \
        --sim-score-log data/memory/esc/sim_score_log.json

    log "[6/7] Merge PFD into memory (ESC)"
    "${PY}" -m src.instruction.add_pfd \
        --memory data/memory/esc/updated.json \
        --pfd data/pfd/esc_pfd.json \
        --output data/memory/esc/updated_pfd.json

    log "[7/7] Build instruction dataset (ESC)"
    "${PY}" -m src.instruction.upd_pfd2inst \
        --input data/memory/esc/updated_pfd.json \
        --train-out data/instruction/esc/train.json \
        --test-out  data/instruction/esc/test.json
}

run_cpsy() {
    log "CPsy pipeline"

    log "[1/7] EToM 7-factor generation (CPsy)"
    "${PY}" -m src.etom.gen_seven_cpsy \
        --input "${CPSY_INPUT}" \
        --output-dir data/etom/cpsy_raw \
        --exemplar-bank config/etom_exemplars_cpsy.json

    log "[2/7] Parse EToM .txt → JSON (CPsy)"
    "${PY}" -m src.etom.parse_seven \
        --input-dir data/etom/cpsy_raw \
        --output-file data/etom/cpsy_seven.json

    log "[3/7] PFD generation (CPsy)"
    "${PY}" -m src.pfd.gen_pfd_cpsy \
        --input data/etom/cpsy_seven.json \
        --output-dir data/pfd/cpsy_raw

    log "[4/7] Parse PFD .txt → JSON (CPsy)"
    "${PY}" -m src.pfd.parse_pfd \
        --style cpsy \
        --input-dir data/pfd/cpsy_raw \
        --output-file data/pfd/cpsy_pfd.json \
        --base-json data/etom/cpsy_seven.json

    log "[5/7] Memory update with ORM (CPsy)"
    mkdir -p data/memory/cpsy
    "${PY}" -m src.memory.run_update \
        --input data/etom/cpsy_seven.json \
        --output data/memory/cpsy/updated.json \
        --similar-log data/memory/cpsy/similar_log.json \
        --outdated-log data/memory/cpsy/outdated_log.json \
        --sim-score-log data/memory/cpsy/sim_score_log.json

    log "[6/7] Merge PFD into memory (CPsy)"
    "${PY}" -m src.instruction.add_pfd \
        --memory data/memory/cpsy/updated.json \
        --pfd data/pfd/cpsy_pfd.json \
        --output data/memory/cpsy/updated_pfd.json

    log "[7/7] Build instruction dataset (CPsy)"
    "${PY}" -m src.instruction.upd_pfd2inst \
        --input data/memory/cpsy/updated_pfd.json \
        --train-out data/instruction/cpsy/train.json \
        --test-out  data/instruction/cpsy/test.json
}

if [ "${RUN_ESC}" = "1" ]; then
    run_esc
fi

if [ "${RUN_CPSY}" = "1" ]; then
    run_cpsy
fi

log "Pipeline done. Outputs under data/{etom,pfd,memory,instruction}/."
