#!/bin/bash
#SBATCH --job-name=flora_tinyllama_wiz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END

set -e

mkdir -p logs

# Step 1: Load conda and activate FLoRA environment
module load conda
conda activate flora

# Step 2: Download TinyLlama
python download.py

# Step 3: Run FLoRA (Homogeneous) - TinyLlama, Wizard, MMLU
# Paper settings: 3 rounds, 1 epoch, rank=16, 10 clients (Table 1 + Appendix Table 2)
python main.py \
  --global_model 'tinyllama' \
  --data_path './data_wiz' \
  --output_dir './flora-tinyllama-homo-wiz/' \
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
  --dev_data_path './mmlu_test_1444.jsonl'

