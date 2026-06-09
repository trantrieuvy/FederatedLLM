---
run_id: 2026-06-02-30286-rte-normal-flora-e20r150-pipeline
created_at: 2026-06-02 11:42 CEST
last_checked: 2026-06-04 12:17 CEST
job_name: "rte_flora_e20r150_s001_r01 ... rte_flora_e20r150_s015_r03"
slurm_job_id: "30286-30330"
array: "single-row arrays, one row per job"
status: active_segment_7
repo: "/homes/neumann/trieu.vy.tran/FederatedLLM"
workdir: "/homes/neumann/trieu.vy.tran/FederatedLLM"
submit_command: "scripts/submit_rte_normal_flora_e20r150_pipeline.sh 1 15"
script: "run_epoch_round_tuning.sh"
submit_helper: "scripts/submit_rte_normal_flora_e20r150_pipeline.sh"
manifest: "tuning_manifests/roberta_rte_stratified_normal_flora_only_seed0_e20_r150.tsv"
manifest_source: explicit
entrypoint: "python main_roberta_glue.py via run_epoch_round_tuning.sh"
key_parameters: "Normal RTE FLoRA, RTE stratified, roberta-base, homo, epochs=20, rounds=150, seed=0, num_clients in {3,10,20}, lora_r=4, lora_alpha=4, local_val_set_size=0.1, local_train_monitor_size=0, ROBERTA_LOCAL_VALIDATION_SOURCE=global_val, ROBERTA_RESUME_FROM_LATEST=False for segment 1 and True for segments 2-15, ROBERTA_MAX_ROUNDS_PER_INVOCATION=10, ROBERTA_RETAIN_ADAPTER_EVERY_N_ROUNDS=0, ROBERTA_LOCAL_MONITOR_ACCURACY=True, FORCE=true for resume segments"
run_root: "./epoch_round_tuning_rte_normal_flora"
log_pattern: "logs/%x_%A_%a.{out,err}"
notes: "Separate clean normal FLoRA run root for the four-method RTE plot comparison; does not overwrite the legacy cumulative flora outputs in epoch_round_tuning_rte_client_count_monitored."
---

# RTE Normal FLoRA e20/r150 Pipeline

## Summary

- Purpose: Run clean normal merged FLoRA for RTE across 3, 10, and 20 clients so the RTE monitored comparison can include four method curves: normal FLoRA, legacy/cumulative Linear FLoRA, Nonlinear FLoRA Cumulative, and Nonlinear FFA.
- Submit command: `scripts/submit_rte_normal_flora_e20r150_pipeline.sh 1 15`.
- Successful job id range: `30286`-`30330`, one single-row array job per `(segment, manifest row)`.
- Run root: `./epoch_round_tuning_rte_normal_flora`.
- Manifest: `tuning_manifests/roberta_rte_stratified_normal_flora_only_seed0_e20_r150.tsv`.

## Manifest Rows

| Row | Method | Dataset | Model | Setting | Epochs | Rounds | Seed | Clients | Rank |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `flora` | `rte_stratified` | `roberta-base` | `homo` | 20 | 150 | 0 | 3 | 4 |
| 2 | `flora` | `rte_stratified` | `roberta-base` | `homo` | 20 | 150 | 0 | 10 | 4 |
| 3 | `flora` | `rte_stratified` | `roberta-base` | `homo` | 20 | 150 | 0 | 20 | 4 |

## Dependency Shape

- Segment size: 10 communication rounds per invocation.
- Total segments: 15, covering rounds 0-149.
- Row-level sequencing: segment `N` row `R` waits for segment `N-1` row `R`.
- Helper lane dependencies preserve the default 8-lane global cap.
- Segment 1 starts fresh with `ROBERTA_RESUME_FROM_LATEST=False`; later segments resume from each row's `server_state.pt`.

## First Status

- `2026-06-02 11:42 CEST`: Submitted jobs `30286`-`30330`.
- `2026-06-02 11:42 CEST`: `squeue --me` showed segment-1 jobs `30286_1`, `30287_2`, and `30288_3` pending for resources/priority, with all later jobs pending on dependencies.
- `2026-06-02 11:42 CEST`: `scontrol show job 30286` confirmed `RUN_ROOT=./epoch_round_tuning_rte_normal_flora`, manifest `roberta_rte_stratified_normal_flora_only_seed0_e20_r150.tsv`, `ROBERTA_RESUME_FROM_LATEST=False`, `ROBERTA_MAX_ROUNDS_PER_INVOCATION=10`, `ROBERTA_LOCAL_MONITOR_ACCURACY=True`, `ROBERTA_LOCAL_VALIDATION_SOURCE=global_val`, and `ROBERTA_LOCAL_TRAIN_MONITOR_SIZE_OVERRIDE=0`.
- `2026-06-02 11:42 CEST`: `scontrol show job 30330` confirmed the final segment dependency chain with `ROBERTA_RESUME_FROM_LATEST=True` and `FORCE=true`.
- `2026-06-02 12:16 CEST`: `30286_1` was running on `gpunode04`; `30287_2` was pending for resources, `30288_3` was pending for priority, and later segment jobs remained dependency-gated.
- `2026-06-04 11:39 CEST`: `squeue --me` showed segment 7 running for all three rows: `30304_1` (`s007_r01`, 3 clients, 1:45 runtime on `gpunode02`), `30305_2` (`s007_r02`, 10 clients, 1:09 runtime on `gpunode06`), and `30306_3` (`s007_r03`, 20 clients, 0:40 runtime on `gpunode02`). Later jobs `30307`-`30330` were dependency-gated. Stdout was advancing at rounds 64, 62, and 61 respectively, and `server_state.pt` timestamps under `epoch_round_tuning_rte_normal_flora/.../seed0/{3,10,20}/` were fresh at 11:24, 11:26, and 11:33. Active stderr logs contained progress bars only; `rg` found no `Traceback`, `ERROR`, `RuntimeError`, OOM, killed, exception, or failed patterns.
- `2026-06-04 12:17 CEST`: Segment 7 was still running for all three rows: `30304_1` at 2:23 elapsed, `30305_2` at 1:47 elapsed, and `30306_3` at 1:18 elapsed. Later jobs `30307`-`30330` remained dependency-gated. Stdout showed row 1 had saved `Acc round 65: 0.6895306859205776` and started round 66; row 2 was in round 63; row 3 was in round 62. `server_state.pt` timestamps under `epoch_round_tuning_rte_normal_flora/.../seed0/{3,10,20}/` refreshed to 12:11, 11:55, and 12:09. Active stderr logs contained progress bars only; `rg` found no `Traceback`, `ERROR`, `RuntimeError`, OOM, killed, exception, or failed patterns.

## Next Checks

- When `s001` starts, confirm stdout logs show `method: flora` with normal merged FLoRA behavior and not legacy cumulative resume behavior.
- After segment 1 completes, confirm each client-count output has `server_state.pt` under `epoch_round_tuning_rte_normal_flora/tuning-flora-rte_stratified-roberta-base-homo-e20-r150/seed0/{3,10,20}/`.
- After completion, update `epoch_round_tuning_analysis.ipynb` so the RTE monitored aggregated validation plot reads both the legacy monitored run root and this clean normal FLoRA run root.
