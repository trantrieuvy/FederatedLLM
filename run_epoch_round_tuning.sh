#!/bin/bash
#SBATCH --job-name=epoch_round_tuning
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

MANIFEST="${MANIFEST:-tuning_manifests/tinyllama_coarse.tsv}"
DEV_DATA_PATH="${DEV_DATA_PATH:-./mmlu_test_1444.jsonl}"
CONDA_ENV="${CONDA_ENV:-flora}"
RUN_ROOT="${RUN_ROOT:-./epoch_round_tuning}"
FORCE="${FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-false}"
MANIFEST_INDEX_BASE="${MANIFEST_INDEX_BASE:-1}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LAYERCRAFT_PATH="${LAYERCRAFT_PATH:-${SCRIPT_DIR}/../layercraft}"

mkdir -p logs
mkdir -p "${RUN_ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

setup_env() {
  if [[ "${SKIP_ENV_SETUP}" == "true" || "${DRY_RUN}" == "true" ]]; then
    return
  fi

  module load conda
  conda activate "${CONDA_ENV}"

  if [[ -d "${LAYERCRAFT_PATH}/layercraft" ]]; then
    export PYTHONPATH="${LAYERCRAFT_PATH}:${PYTHONPATH:-}"
    echo "Using layercraft from ${LAYERCRAFT_PATH}"
  else
    echo "Layercraft path not found: ${LAYERCRAFT_PATH}" >&2
  fi
}

model_arg_for() {
  local model="$1"
  case "${model}" in
    tinyllama)
      echo "tinyllama"
      ;;
    llama|llama-7b)
      echo "llama-7b"
      ;;
    roberta|roberta-base)
      echo "roberta-base"
      ;;
    *)
      echo "Unknown model: ${model}" >&2
      return 1
      ;;
  esac
}

model_tag_for() {
  local model="$1"
  case "${model}" in
    tinyllama)
      echo "tinyllama"
      ;;
    llama|llama-7b)
      echo "llama"
      ;;
    roberta|roberta-base)
      echo "roberta-base"
      ;;
    *)
      echo "Unknown model: ${model}" >&2
      return 1
      ;;
  esac
}

log_path_for() {
  local method="$1"
  local output_dir="$2"

  if [[ "${method}" == "flora" ]]; then
    echo "${output_dir}10log.txt"
  elif [[ "${method}" == "ffa" || "${method}" == "nonlinear_flora" ]]; then
    echo "${output_dir}10/log.txt"
  else
    echo "Unknown method: ${method}" >&2
    return 1
  fi
}

glue_task_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    rte|rte_stratified|rte_dirichlet)
      echo "rte"
      ;;
    *)
      echo "${dataset}"
      ;;
  esac
}

run_row() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  local setting="$4"
  local epochs="$5"
  local rounds="$6"
  local seed="$7"

  local model_arg
  local model_tag
  model_arg="$(model_arg_for "${model}")"
  model_tag="$(model_tag_for "${model}")"

  local data_path="./data_${dataset}"
  local output_dir="${RUN_ROOT%/}/tuning-${method}-${dataset}-${model_tag}-${setting}-e${epochs}-r${rounds}/seed${seed}/"
  local log_path
  log_path="$(log_path_for "${method}" "${output_dir}")"

  if [[ ! -d "${data_path}/10" ]]; then
    echo "Data path does not contain 10-client split: ${data_path}/10" >&2
    exit 1
  fi

  if [[ "${FORCE}" != "true" && -s "${log_path}" ]]; then
    echo "Skipping completed run: ${output_dir}"
    return
  fi

  local heter_flag="False"
  if [[ "${setting}" == "heter" ]]; then
    heter_flag="True"
  elif [[ "${setting}" != "homo" ]]; then
    echo "Unknown setting: ${setting}" >&2
    exit 1
  fi

  local cmd=()
  if [[ "${model_arg}" == "roberta-base" ]]; then
    local glue_task
    glue_task="$(glue_task_for_dataset "${dataset}")"
    cmd=(
      python main_roberta_glue.py
      --method "${method}"
      --global_model "${model_arg}"
      --task_name "${glue_task}"
      --data_path "${data_path}"
      --output_dir "${output_dir}"
      --num_communication_rounds "${rounds}"
      --num_clients 10
      --local_num_epochs "${epochs}"
      --local_batch_size 32
      --local_micro_batch_size 16
      --local_learning_rate 5e-4
      --warmup_ratio 0.06
      --weight_decay 0.1
      --max_seq_length 512
      --lora_r 8
      --lora_alpha 8
      --heter "${heter_flag}"
      --seed "${seed}"
    )
  else
    case "${method}" in
      flora)
        cmd=(
          python main.py
          --global_model "${model_arg}"
          --data_path "${data_path}"
          --output_dir "${output_dir}"
          --num_communication_rounds "${rounds}"
          --num_clients 10
          --local_num_epochs "${epochs}"
          --local_batch_size 128
          --local_micro_batch_size 16
          --local_learning_rate 3e-4
          --lora_r 16
          --lora_alpha 32
          --stacking True
          --heter "${heter_flag}"
          --full False
          --dev_data_path "${DEV_DATA_PATH}"
          --seed "${seed}"
        )
        ;;
      nonlinear_flora)
        cmd=(
          python main_nonlinear_flora.py
          --global_model "${model_arg}"
          --data_path "${data_path}"
          --output_dir "${output_dir}"
          --num_communication_rounds "${rounds}"
          --num_clients 10
          --local_num_epochs "${epochs}"
          --local_batch_size 128
          --local_micro_batch_size 16
          --local_learning_rate 3e-4
          --lora_r 16
          --lora_alpha 32
          --heter "${heter_flag}"
          --dev_data_path "${DEV_DATA_PATH}"
          --seed "${seed}"
        )
        ;;
      ffa)
        cmd=(
          python main_ffa.py
          --global_model "${model_arg}"
          --data_path "${data_path}"
          --output_dir "${output_dir}"
          --num_communication_rounds "${rounds}"
          --num_clients 10
          --local_num_epochs "${epochs}"
          --local_batch_size 128
          --local_micro_batch_size 16
          --local_learning_rate 3e-4
          --lora_r 16
          --lora_alpha 32
          --activation gelu
          --heter "${heter_flag}"
          --dev_data_path "${DEV_DATA_PATH}"
          --seed "${seed}"
        )
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ "${setting}" == "heter" ]]; then
    if [[ "${model_arg}" == "roberta-base" ]]; then
      cmd+=(--local_ranks "[32,16,8,8,4,4,2,2,2,2]")
    else
      cmd+=(--local_ranks "[64,32,16,16,8,8,4,4,4,4]")
    fi
  fi

  echo "============================================"
  echo "method=${method} dataset=${dataset} model=${model_tag} setting=${setting}"
  echo "epochs=${epochs} rounds=${rounds} seed=${seed}"
  echo "output_dir=${output_dir}"
  echo "dev_data_path=${DEV_DATA_PATH}"
  echo "============================================"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi

  "${cmd[@]}"
}

setup_env

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  line_number=$((SLURM_ARRAY_TASK_ID + 2 - MANIFEST_INDEX_BASE))
  row="$(sed -n "${line_number}p" "${MANIFEST}")"
  if [[ -z "${row}" ]]; then
    echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  IFS=$'\t' read -r method dataset model setting epochs rounds seed <<< "${row}"
  run_row "${method}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}"
else
  while IFS=$'\t' read -r method dataset model setting epochs rounds seed; do
    [[ -z "${method}" || "${method}" == "method" ]] && continue
    run_row "${method}" "${dataset}" "${model}" "${setting}" "${epochs}" "${rounds}" "${seed}"
  done < "${MANIFEST}"
fi
