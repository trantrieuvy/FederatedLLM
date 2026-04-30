#!/bin/bash
#SBATCH --job-name=ffa_dolly_stratified_seed0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --nodelist=gpunode06
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

set -e

mkdir -p logs

# Step 1: Load conda and activate FLoRA environment
module load conda
conda activate flora

SEEDS=(0)
DATASET="${DATASET:-dolly_stratified}"
DATA_ROOT="${DATA_ROOT:-./data_${DATASET}}"
DATA_PATH="${DATA_ROOT}"
OUTPUT_TAG="${OUTPUT_TAG:-$(basename "${DATA_ROOT}" | sed 's/^data_//')}"

for SEED in "${SEEDS[@]}"; do
  echo "============================================"
  echo "Running all FFA experiments with seed=${SEED}"
  echo "Dataset root: ${DATA_PATH}"
  echo "============================================"

  # TinyLlama Homogeneous
  echo "[seed=${SEED}] TinyLlama FFA Homo"
  python main_ffa.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./ffa-tinyllama-homo-${OUTPUT_TAG}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --activation 'gelu' \
    --heter False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # TinyLlama Heterogeneous
  echo "[seed=${SEED}] TinyLlama FFA Heter"
  python main_ffa.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./ffa-tinyllama-heter-${OUTPUT_TAG}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --activation 'gelu' \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # Llama-7B Homogeneous
  echo "[seed=${SEED}] Llama-7B FFA Homo"
  python main_ffa.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./ffa-llama-homo-${OUTPUT_TAG}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --activation 'gelu' \
    --heter False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # Llama-7B Heterogeneous
  echo "[seed=${SEED}] Llama-7B FFA Heter"
  python main_ffa.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./ffa-llama-heter-${OUTPUT_TAG}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --activation 'gelu' \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}
done
