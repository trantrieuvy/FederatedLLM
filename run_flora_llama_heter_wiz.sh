#!/bin/bash
#SBATCH --job-name=flora_llama_heter_wiz
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

# Step 2: Run FLoRA (Heterogeneous) - Llama-7B, Wizard, MMLU
# Paper settings: 1 round, 1 epoch, heterogeneous ranks [64,32,16,16,8,8,4,4,4,4], 10 clients
# Target: Table 1 Llama Heter FLoRA Wizard MMLU = 27.91
python main.py \
  --global_model 'llama-7b' \
  --data_path './data_wiz' \
  --output_dir './flora-llama-heter-wiz/' \
  --num_communication_rounds 1 \
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
  --dev_data_path './mmlu_test_1444.jsonl'
