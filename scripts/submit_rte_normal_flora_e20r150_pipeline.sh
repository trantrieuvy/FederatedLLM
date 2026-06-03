#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export REPO_DIR
export MANIFEST="${MANIFEST:-tuning_manifests/roberta_rte_stratified_normal_flora_only_seed0_e20_r150.tsv}"
export RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning_rte_normal_flora}"
export JOB_PREFIX="${JOB_PREFIX:-rte_flora_e20r150}"
export ROBERTA_LOCAL_MONITOR_ACCURACY="${ROBERTA_LOCAL_MONITOR_ACCURACY:-True}"
export ROBERTA_LOCAL_VALIDATION_SOURCE="${ROBERTA_LOCAL_VALIDATION_SOURCE:-global_val}"
export ROBERTA_LOCAL_TRAIN_MONITOR_SIZE_OVERRIDE="${ROBERTA_LOCAL_TRAIN_MONITOR_SIZE_OVERRIDE:-0}"
export ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-0}"

exec "${REPO_DIR}/scripts/submit_rte_e20r150_pipeline.sh" "$@"
