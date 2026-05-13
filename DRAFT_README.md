# Federated Adapter Composition for LLM Fine-Tuning

This project studies federated instruction tuning with heterogeneous, cumulative, and nonlinear low-rank adapters. It builds on FLoRA-style federated aggregation and extends the setup with controlled data partitioning, LayerCraft adapter variants, and multi-round residual adapter composition.

## Highlights

- Federated fine-tuning for instruction-following LLMs with LoRA-style adapters.
- Heterogeneous client ranks for non-IID federated settings.
- Cumulative linear FLoRA, where each round adds a fresh residual adapter.
- Nonlinear FLoRA, where each residual adapter uses `B * sigma(A * x)`.
- LayerCraft integration for swapping adapter implementations while preserving the federated training loop.
- Dolly and WizardLM client split utilities, including stratified client allocation.

## Repository Layout

```text
main.py                         # PEFT/FLoRA baseline
main_layercraft.py              # LayerCraft-backed adapter experiments
main_linear_flora_cumulative.py # cumulative linear residual adapters
main_nonlinear_flora.py         # cumulative nonlinear residual adapters
main_ffa.py                     # FFA comparison experiments
client_data_allocation.py       # Dolly/Wizard federated data splits
fed_utils/                      # clients, aggregation, scheduling, evaluation
utils/                          # prompting, schema helpers, analysis helpers
templates/                      # prompt templates
run_*.sh                        # Slurm/local experiment launchers
```

## Installation

```bash
conda create -n flora python=3.10
conda activate flora
pip install -r requirements.txt
```

For MMLU evaluation, install `lm-evaluation-harness` according to its upstream instructions.

## Data Preparation

Generated datasets are not committed by default. Create federated splits with:

```bash
python client_data_allocation.py \
  --dataset dolly \
  --num-clients 10 \
  --mode dirichlet \
  --output-root data_dolly

python client_data_allocation.py \
  --dataset dolly \
  --num-clients 10 \
  --mode stratified_keep_sizes \
  --source-root data_dolly \
  --output-root data_dolly_stratified
```

Each split writes a `split_metadata.json` file describing the split policy, seed, holdout policy, and client-size policy.

## Quickstart

Run a PEFT/FLoRA baseline:

```bash
python main.py \
  --global_model tinyllama \
  --data_path ./data_wiz \
  --output_dir ./runs/flora-tinyllama-homo-wiz/seed0/ \
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
  --dev_data_path ./mmlu_test_1444.jsonl \
  --seed 0
```

Run nonlinear multi-round FLoRA:

```bash
python main_nonlinear_flora.py \
  --global_model tinyllama \
  --data_path ./data_wiz \
  --output_dir ./runs/nonlinear-flora-tinyllama-homo-wiz/seed0/ \
  --num_communication_rounds 3 \
  --num_clients 10 \
  --local_num_epochs 1 \
  --local_batch_size 128 \
  --local_micro_batch_size 16 \
  --local_learning_rate 3e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --dev_data_path ./mmlu_test_1444.jsonl \
  --seed 0
```

Run LayerCraft verification:

```bash
bash run_layercraft_verify.sh
```

## Experiments

- Experiment 1: expressivity comparison between baseline linear FLoRA, doubled-rank linear adapters, and nonlinear adapters.
- Experiment 2: multi-round nonlinear FLoRA with fresh residual adapters per communication round.
- Epoch/round tuning: compare local epoch count and communication-round tradeoffs.
- Data split comparison: legacy Dirichlet splits versus stratified client-preserving splits.

## Results

Add a compact results table here once the final experiment set is decided.

Recommended columns:

- Dataset
- Model
- Method
- Rank setting
- Rounds
- Seed count
- MMLU or task score
- Notes

Figures can live under `assets/figures/`.

## Attribution

This work builds on FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations. Portions of the baseline training flow are adapted from the original FLoRA implementation. The adapter experiments also use PEFT, Hugging Face Transformers, and LayerCraft.

## License

Add the license after checking the upstream FLoRA repository license and any LayerCraft licensing requirements.

