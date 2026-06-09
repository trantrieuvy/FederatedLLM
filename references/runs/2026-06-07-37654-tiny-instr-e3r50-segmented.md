---
run_id: 2026-06-07-37654-tiny-instr-e3r50-segmented
date: 2026-06-07
job_name: "tiny_instr_e3r50seg_s001_r01 ... tiny_instr_e3r50seg_s005_r24"
slurm_job_id: "37654-37773"
status: "active"
last_checked: "2026-06-07 14:28 CEST"
repo: "/homes/neumann/trieu.vy.tran/FederatedLLM"
submit_command: "CAP=8 scripts/submit_tinyllama_instruction_e3r50_pipeline.sh 1 5"
submit_helper: "scripts/submit_tinyllama_instruction_e3r50_pipeline.sh"
manifest: "tuning_manifests/tinyllama_dolly_wizard_stratified_client_count_seed0_e3_r50.tsv"
array: "single-row arrays, one job per segment and manifest row"
stdout: "logs/%x_%A_%a.out"
stderr: "logs/%x_%A_%a.err"
run_root: "./epoch_round_tuning_instruction_client_count_e3r50_segmented"
---

# TinyLLaMA Stratified Dolly/Wizard e3/r50 Segmented Pipeline

## Summary

- Purpose: Clean rerun of the TinyLLaMA stratified Dolly/Wizard client-count sweep after cancelling pre-resume job `37358`.
- Scope: Linear FLoRA (`flora`) and Nonlinear FFA (`ffa`), Dolly stratified and Wizard stratified, Homo/Heter, client counts `3`, `10`, and `20`, seed `0`.
- Manifest rows: `1-24` from `tuning_manifests/tinyllama_dolly_wizard_stratified_client_count_seed0_e3_r50.tsv`.
- Job ids: `37654` through `37773`, submitted by `CAP=8 scripts/submit_tinyllama_instruction_e3r50_pipeline.sh 1 5`.
- Run root: `./epoch_round_tuning_instruction_client_count_e3r50_segmented`.

## Segmentation

- Total configured communication rounds per manifest row: `50`.
- Segment size: `10` rounds per Slurm invocation.
- Total segments: `5`.
- Segment 1 exports `TINYLLAMA_RESUME_FROM_LATEST=False`, `TINYLLAMA_MAX_ROUNDS_PER_INVOCATION=10`, and `TINYLLAMA_RETAIN_ADAPTER_EVERY_N_ROUNDS=1`.
- Segments 2-5 export `TINYLLAMA_RESUME_FROM_LATEST=True`, `TINYLLAMA_MAX_ROUNDS_PER_INVOCATION=10`, and `TINYLLAMA_RETAIN_ADAPTER_EVERY_N_ROUNDS=1`.
- Each `(segment,row)` job is a single-row Slurm array whose array index equals the manifest row.
- Dependencies include per-row `afterok` dependencies across segments plus a lane dependency chain so the helper-submitted jobs are capped at `CAP=8`.

## Checkpoint Behavior

- TinyLLaMA Linear FLoRA now writes committed per-round adapter artifacts plus `round_config.json` and `server_state.pt`, and later invocations resume by replaying committed adapters.
- TinyLLaMA FFA now writes `round_config.json` and `server_state.pt` with the global B state, and later invocations resume from that state.
- Final `log.txt` score histories are written only after all `50` rounds complete. The notebook can also read committed partial round scores from `round_config.json`.
- The Linear FLoRA heter stacked-rank fix is included for this clean run.

## Job Map

- Segment 1: `37654`-`37677`, names `tiny_instr_e3r50seg_s001_r01` ... `tiny_instr_e3r50seg_s001_r24`.
- Segment 2: `37678`-`37701`, names `tiny_instr_e3r50seg_s002_r01` ... `tiny_instr_e3r50seg_s002_r24`.
- Segment 3: `37702`-`37725`, names `tiny_instr_e3r50seg_s003_r01` ... `tiny_instr_e3r50seg_s003_r24`.
- Segment 4: `37726`-`37749`, names `tiny_instr_e3r50seg_s004_r01` ... `tiny_instr_e3r50seg_s004_r24`.
- Segment 5: `37750`-`37773`, names `tiny_instr_e3r50seg_s005_r01` ... `tiny_instr_e3r50seg_s005_r24`.

## Status Checks

- `2026-06-07 14:28 CEST`: Submitted `120` single-row array jobs, job ids `37654`-`37773`.
- `2026-06-07 14:28 CEST`: `squeue --me` showed segment-1 rows `1` and `2` running on `gpunode06`; segment-1 rows `3`-`8` pending with `QOSMaxGRESPerUser`; segment-1 rows `9`-`24` dependency-gated; later segments dependency-gated.
- `2026-06-07 14:28 CEST`: Stdout for `37654_1` and `37655_2` confirmed `output_dir=./epoch_round_tuning_instruction_client_count_e3r50_segmented/...`, `resume_from_latest=False`, `max_rounds_per_invocation=10`, and `retain_adapter_every_n_rounds=1`.
- `2026-06-07 14:28 CEST`: Notebook loader sanity check against the segmented run root reported `24` status rows, `0` score rows, and all rows as `No scores`, confirming it does not inherit stale stdout from cancelled job `37358`.
- `2026-06-07 15:40 CEST`: Refreshed `epoch_round_tuning_analysis.ipynb` so the "Stratified Dolly/Wizard e3/r50 Client-Count Runs" section uses committed segmented `round_config.json` accuracy as the primary source and only merges matching `tiny_instr_e3r50seg_*` stdout for newer in-progress rounds. The refreshed section loaded `5` segmented round-score rows from `round_config`, found `3` committed partial schedules, wrote `tuning_analysis_outputs/instruction_client_count_e3r50/{run_status,round_scores,round_summary}.csv`, and wrote current HTML plots under `plots_epoch_round_tuning/instruction_client_count_e3r50/`.

## Next Checks

- Watch `logs/tiny_instr_e3r50seg_s001_r01_37654_1.out` and `logs/tiny_instr_e3r50seg_s001_r02_37655_2.out` for first round completion and committed `round_config.json`.
- After any segment-1 row completes, verify its segment-2 row resumes from committed state rather than starting at communication round `0`.
- Watch rows `15`, `18`, `21`, and `24` for memory pressure because they are the 20-client Wizard rows.
