#!/bin/bash
#SBATCH --job-name=layercraft_verify
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

set -e

mkdir -p logs

# Step 1: Load conda and activate environment
module load conda
conda activate flora

# ==============================================================================
# Layercraft verification: mirror run_all_seeds.sh settings with seed=0
#
# Goal: compare accuracy with the PEFT baseline (run_all_seeds.sh) to verify
# that layercraft is a correct replication of the plain LoRA setup.
#
# Settings are identical to run_all_seeds.sh except:
#   - Uses main_layercraft.py instead of main.py
#   - No --full flag (layercraft is adapter-only)
#   - Output dirs prefixed with "layercraft-" to avoid overwriting PEFT results
#
# NOTE: Results will be close but NOT bit-identical because:
#   - PEFT initializes A with kaiming_uniform; layercraft uses normal(0, 0.02)
#   - Both initialize B with zeros, and both freeze the base model
#   - The forward pass (W + alpha/r * B @ A) is the same
#   - Same optimizer, same data, same seed — so the training dynamics are similar
# ==============================================================================

SEED=0

echo "============================================"
echo "Layercraft verification — seed=${SEED}"
echo "============================================"

# TinyLlama Homogeneous
echo "[seed=${SEED}] TinyLlama Homo"
python main_layercraft.py \
  --global_model 'tinyllama' \
  --data_path './data_wiz' \
  --output_dir "./layercraft-tinyllama-homo-wiz/seed${SEED}/" \
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
  --dev_data_path './mmlu_test_1444.jsonl' \
  --seed ${SEED}

# TinyLlama Heterogeneous
# echo "[seed=${SEED}] TinyLlama Heter"
# python main_layercraft.py \
#   --global_model 'tinyllama' \
#   --data_path './data_wiz' \
#   --output_dir "./layercraft-tinyllama-heter-wiz/seed${SEED}/" \
#   --num_communication_rounds 3 \
#   --num_clients 10 \
#   --local_num_epochs 1 \
#   --local_batch_size 128 \
#   --local_micro_batch_size 16 \
#   --local_learning_rate 3e-4 \
#   --lora_r 16 \
#   --stacking True \
#   --heter True \
#   --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
#   --dev_data_path './mmlu_test_1444.jsonl' \
#   --seed ${SEED}

# Llama-7B Homogeneous
# echo "[seed=${SEED}] Llama-7B Homo"
# python main_layercraft.py \
#   --global_model 'llama-7b' \
#   --data_path './data_wiz' \
#   --output_dir "./layercraft-llama-homo-wiz/seed${SEED}/" \
#   --num_communication_rounds 1 \
#   --num_clients 10 \
#   --local_num_epochs 1 \
#   --local_batch_size 128 \
#   --local_micro_batch_size 16 \
#   --local_learning_rate 3e-4 \
#   --lora_r 16 \
#   --lora_alpha 32 \
#   --stacking True \
#   --heter False \
#   --dev_data_path './mmlu_test_1444.jsonl' \
#   --seed ${SEED}

# Llama-7B Heterogeneous
# echo "[seed=${SEED}] Llama-7B Heter"
# python main_layercraft.py \
#   --global_model 'llama-7b' \
#   --data_path './data_wiz' \
#   --output_dir "./layercraft-llama-heter-wiz/seed${SEED}/" \
#   --num_communication_rounds 1 \
#   --num_clients 10 \
#   --local_num_epochs 1 \
#   --local_batch_size 128 \
#   --local_micro_batch_size 16 \
#   --local_learning_rate 3e-4 \
#   --lora_r 16 \
#   --stacking True \
#   --heter True \
#   --local_ranks '[64,32,16,16,8,8,4,4,4,4]' \
#   --dev_data_path './mmlu_test_1444.jsonl' \
#   --seed ${SEED}

echo "============================================"
echo "Done. Compare results with PEFT baseline:"
echo "  PEFT:       flora-{tinyllama,llama}-{homo,heter}-wiz/seed${SEED}/log.txt"
echo "  Layercraft: layercraft-{tinyllama,llama}-{homo,heter}-wiz/seed${SEED}/log.txt"
echo "============================================"
