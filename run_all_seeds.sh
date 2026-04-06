#!/bin/bash
#SBATCH --job-name=flora_dolly_seed2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --nodelist=gpunode06
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

set -e

mkdir -p logs

# Step 1: Load conda and activate FLoRA environment
module load conda
conda activate flora

SEEDS=(2)
DATASET="dolly"
DATA_PATH="./data_${DATASET}"  # Dolly is split into ./data/10/ by client_data_allocation.py

for SEED in "${SEEDS[@]}"; do
  echo "============================================"
  echo "Running all experiments with seed=${SEED}"
  echo "============================================"

  # TinyLlama Homogeneous
  echo "[seed=${SEED}] TinyLlama Homo"
  python main.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./flora-tinyllama-homo-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --stacking True \
    --heter False \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # TinyLlama Heterogeneous
  echo "[seed=${SEED}] TinyLlama Heter"
  python main.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./flora-tinyllama-heter-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --stacking True \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # Llama-7B Homogeneous
  echo "[seed=${SEED}] Llama-7B Homo"
  python main.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./flora-llama-homo-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 3 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --stacking True \
    --heter False \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # Llama-7B Heterogeneous
  echo "[seed=${SEED}] Llama-7B Heter"
  python main.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./flora-llama-heter-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 3 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --stacking True \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

done