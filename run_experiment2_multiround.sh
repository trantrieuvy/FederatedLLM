#!/bin/bash
#SBATCH --job-name=exp2_multiround_nonlinear
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=gpunode06
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

# Experiment 2 — Multi-Round Nonlinear FLoRA
#
# Each round clients train a fresh nonlinear adapter B·σ(A·x) on top of
# the frozen stacked adapter from the previous round.  After N rounds:
#
#   y = W0·x  +  B0·σ(A0·x)  +  B1·σ(A1·x)  +  …  +  BN·σ(AN·x)
#
# Mirrors run_all_seeds.sh setup (same seeds, dataset, hyperparams) so that
# results are directly comparable to the linear FLoRA baseline.

set -e
mkdir -p logs

module load conda
conda activate flora

SEEDS=(2)
DATASET="wiz"
DATA_PATH="./data_${DATASET}"

for SEED in "${SEEDS[@]}"; do
  echo "============================================"
  echo "Experiment 2  seed=${SEED}"
  echo "============================================"

  # --------------------------------------------------------------------------
  # TinyLlama — multi-round nonlinear FLoRA
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] TinyLlama Nonlinear multi-round homo"
  python main_nonlinear_flora.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp2-tinyllama-nonlinear-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # TinyLlama — multi-round nonlinear FLoRA (heter)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] TinyLlama Nonlinear multi-round heter"
  python main_nonlinear_flora.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp2-tinyllama-nonlinear-3round-heter-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # Llama-7B — single-round nonlinear FLoRA (homo)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] Llama-7B Nonlinear single-round homo"
  python main_nonlinear_flora.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp2-llama-nonlinear-1round-homo-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --heter False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # Llama-7B — single-round nonlinear FLoRA (heter)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] Llama-7B Nonlinear single-round heter"
  python main_nonlinear_flora.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp2-llama-nonlinear-1round-heter-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --heter True \
    --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

done