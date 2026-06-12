#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export REPO_DIR
export MANIFEST="${MANIFEST:-tuning_manifests/roberta_qnli_stratified_flora_ffa_rank4_seed0_e20_r30.tsv}"
export RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning_qnli_client_count_e20_r30}"
export JOB_PREFIX="${JOB_PREFIX:-qnli_e20r30}"
export TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-3}"
export SEGMENT_ROUNDS="${SEGMENT_ROUNDS:-10}"
export SBATCH_MEM="${SBATCH_MEM:-64G}"
export ROBERTA_LOCAL_MONITOR_ACCURACY="${ROBERTA_LOCAL_MONITOR_ACCURACY:-False}"
export ROBERTA_LOCAL_VALIDATION_SOURCE="${ROBERTA_LOCAL_VALIDATION_SOURCE:-local_holdout}"
export ROBERTA_LOCAL_TRAIN_MONITOR_SIZE_OVERRIDE="${ROBERTA_LOCAL_TRAIN_MONITOR_SIZE_OVERRIDE:-0}"
export ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-0}"

exec "${REPO_DIR}/scripts/submit_rte_e20r150_pipeline.sh" "$@"
