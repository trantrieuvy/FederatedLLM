#!/bin/bash
#SBATCH --job-name=exp1_expressivity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --nodelist=gpunode06
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

# Experiment 1 — Expressivity Comparison
#
# Mirrors run_all_seeds.sh, adding two new conditions per model/setting:
#   - Linear FLoRA r×2  (homo: r=32; heter: [128,64,32,32,16,16,8,8,8,8])
#   - Nonlinear FLoRA r  B·σ(A·x), homo only (main_nonlinear_flora.py)
#
# Existing r=16 linear results are in flora-{model}-{homo,heter}-wiz/seed*/
# Question: does nonlinearity match or beat doubling the rank?

set -e
mkdir -p logs

module load conda
conda activate flora

SEEDS=(2)
DATASET="wiz"
DATA_PATH="./data_${DATASET}"

for SEED in "${SEEDS[@]}"; do
  echo "============================================"
  echo "Experiment 1  seed=${SEED}"
  echo "============================================"

  # --------------------------------------------------------------------------
  # TinyLlama — Linear r=32 (homo)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] TinyLlama Homo Linear r=32"
  python main.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp1-tinyllama-homo-linear-r32-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --stacking True \
    --heter False \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # TinyLlama — Linear doubled ranks (heter)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] TinyLlama Heter Linear doubled ranks"
  python main.py \
    --global_model 'tinyllama' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp1-tinyllama-heter-linear-r2x-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 3 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 32 \
    --stacking True \
    --heter True \
    --local_ranks '[128,64,32,32,16,16,8,8,8,8]' \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # Llama-7B — Linear r=32 (homo)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] Llama-7B Homo Linear r=32"
  python main.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp1-llama-homo-linear-r32-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --stacking True \
    --heter False \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

  # --------------------------------------------------------------------------
  # Llama-7B — Linear doubled ranks (heter)
  # --------------------------------------------------------------------------
  echo "[seed=${SEED}] Llama-7B Heter Linear doubled ranks"
  python main.py \
    --global_model 'llama-7b' \
    --data_path "${DATA_PATH}" \
    --output_dir "./exp1-llama-heter-linear-r2x-${DATASET}/seed${SEED}/" \
    --num_communication_rounds 1 \
    --num_clients 10 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --local_learning_rate 3e-4 \
    --lora_r 32 \
    --stacking True \
    --heter True \
    --local_ranks '[128,64,32,32,16,16,8,8,8,8]' \
    --full False \
    --dev_data_path './mmlu_test_1444.jsonl' \
    --seed ${SEED}

done
