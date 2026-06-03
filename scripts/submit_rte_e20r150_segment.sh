#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_rte_e20r150_segment.sh SEGMENT [--dry-run]
  scripts/submit_rte_e20r150_segment.sh all [--dry-run]

SEGMENT is 1 through 15. Each segment runs 10 communication rounds for all
9 rows in the RTE e20/r150 manifest, with at most 3 array tasks active.

Examples:
  scripts/submit_rte_e20r150_segment.sh 1
  scripts/submit_rte_e20r150_segment.sh 2 --dry-run
  scripts/submit_rte_e20r150_segment.sh all

Environment overrides:
  ARRAY_SPEC=1-9%8
  SEGMENT_ROUNDS=10
  TOTAL_SEGMENTS=15
  RUN_ROOT=./epoch_round_tuning_rte_client_count_monitored
  MANIFEST=tuning_manifests/roberta_rte_stratified_client_count_monitored_rank4_seed0_e20_r150.tsv
  SBATCH_TIME=2-00:00:00
  ROBERTA_LOCAL_MONITOR_ACCURACY=True  # default for segments 3+
USAGE
}

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST="${MANIFEST:-tuning_manifests/roberta_rte_stratified_client_count_monitored_rank4_seed0_e20_r150.tsv}"
RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning_rte_client_count_monitored}"
ARRAY_SPEC="${ARRAY_SPEC:-1-9%8}"
SEGMENT_ROUNDS="${SEGMENT_ROUNDS:-10}"
TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-15}"
RETAIN_ADAPTER_EVERY_N_ROUNDS="${ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS:-0}"
DRY_RUN="false"

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SEGMENT_ARG="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

submit_segment() {
  local segment="$1"
  local dependency_job_id="${2:-}"

  if (( segment < 1 || segment > TOTAL_SEGMENTS )); then
    echo "Segment must be between 1 and ${TOTAL_SEGMENTS}; got ${segment}." >&2
    exit 2
  fi

  local resume="True"
  local force_export=",FORCE=true"
  if (( segment == 1 )); then
    resume="False"
    force_export=""
  fi

  local job_name
  job_name="$(printf 'rte_e20r150_s%03d' "${segment}")"
  local local_monitor_accuracy="${ROBERTA_LOCAL_MONITOR_ACCURACY:-}"
  if [[ -z "${local_monitor_accuracy}" ]]; then
    local_monitor_accuracy="False"
    if (( segment >= 3 )); then
      local_monitor_accuracy="True"
    fi
  fi
  local export_vars
  export_vars="ALL,MANIFEST=${MANIFEST},RUN_ROOT=${RUN_ROOT},ROBERTA_RESUME_FROM_LATEST=${resume},ROBERTA_MAX_ROUNDS_PER_INVOCATION=${SEGMENT_ROUNDS},ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=${RETAIN_ADAPTER_EVERY_N_ROUNDS},ROBERTA_LOCAL_MONITOR_ACCURACY=${local_monitor_accuracy}${force_export}"

  local cmd=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--array=${ARRAY_SPEC}"
    "--export=${export_vars}"
  )

  if [[ -n "${SBATCH_TIME:-}" ]]; then
    cmd+=("--time=${SBATCH_TIME}")
  fi
  if [[ -n "${dependency_job_id}" ]]; then
    cmd+=("--dependency=afterok:${dependency_job_id}")
  fi

  cmd+=(run_epoch_round_tuning.sh)

  cd "${REPO_DIR}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    local job_id
    job_id="$("${cmd[@]}")"
    echo "${job_name}: submitted ${job_id}" >&2
    printf '%s\n' "${job_id}"
  fi
}

if [[ "${SEGMENT_ARG}" == "all" ]]; then
  previous_job_id=""
  for segment in $(seq 1 "${TOTAL_SEGMENTS}"); do
    if [[ "${DRY_RUN}" == "true" ]]; then
      submit_segment "${segment}" "${previous_job_id}"
      previous_job_id="$(printf 'JOBID_s%03d' "${segment}")"
    else
      previous_job_id="$(submit_segment "${segment}" "${previous_job_id}")"
    fi
  done
else
  if ! [[ "${SEGMENT_ARG}" =~ ^[0-9]+$ ]]; then
    echo "SEGMENT must be 1-${TOTAL_SEGMENTS} or all; got ${SEGMENT_ARG}." >&2
    usage
    exit 2
  fi
  submit_segment "${SEGMENT_ARG}"
  echo
fi
