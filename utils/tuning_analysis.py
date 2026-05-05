"""Helpers for controlled local-epoch/communication-round tuning analysis."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px


MANIFEST_COLUMNS = ["method", "dataset", "model", "setting", "epochs", "rounds", "seed"]

RUN_DIR_RE = re.compile(
    r"^tuning-(?P<method>flora|nonlinear_flora|ffa)-(?P<dataset>.+)-"
    r"(?P<model>tinyllama|llama|llama-7b)-(?P<setting>homo|heter)-"
    r"e(?P<epochs>\d+)-r(?P<rounds>\d+)$"
)
LIVE_OUTPUT_DIR_RE = re.compile(r"^output_dir=(?P<output_dir>.+)$", re.MULTILINE)
LIVE_ACCURACY_RE = re.compile(
    r"Acc round\s+(?P<round>\d+):\s+"
    r"(?P<accuracy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

EXP2_NONLINEAR_FLORA_DIR_RE = re.compile(
    r"^exp2-(?P<model>tinyllama|llama)-nonlinear-"
    r"(?:(?P<rounds>\d+)round-)?(?:(?P<setting>heter|homo)-)?(?P<dataset>.+)$"
)

METHOD_LABELS = {
    "flora": "Linear FLoRA",
    "nonlinear_flora": "Nonlinear FLoRA",
    "ffa": "Nonlinear FFA",
}

MODEL_LABELS = {
    "tinyllama": "TinyLlama",
    "llama": "Llama-7B",
    "llama-7b": "Llama-7B",
}

SETTING_LABELS = {
    "homo": "Homo",
    "heter": "Heter",
}

DATASET_LABELS = {
    "wiz": "Wizard",
    "wiz_stratified": "Wizard stratified",
    "dolly": "Dolly",
    "dolly_stratified": "Dolly stratified",
}

PAPER_BASELINES = {
    ("wiz", "tinyllama", "homo"): 43.87,
    ("wiz", "tinyllama", "heter"): 41.48,
    ("wiz", "llama", "homo"): 34.26,
    ("wiz", "llama", "heter"): 27.91,
    ("dolly", "tinyllama", "homo"): 30.80,
    ("dolly", "tinyllama", "heter"): 18.45,
    ("dolly", "llama", "homo"): 30.99,
    ("dolly", "llama", "heter"): 28.50,
    ("dolly_stratified", "tinyllama", "homo"): 30.80,
    ("dolly_stratified", "tinyllama", "heter"): 18.45,
    ("dolly_stratified", "llama", "homo"): 30.99,
    ("dolly_stratified", "llama", "heter"): 28.50,
}

PAPER_BASELINE_ROUNDS = {
    ("wiz", "tinyllama", "homo"): 3,
    ("wiz", "tinyllama", "heter"): 3,
    ("wiz", "llama", "homo"): 1,
    ("wiz", "llama", "heter"): 1,
    ("dolly", "tinyllama", "homo"): 3,
    ("dolly", "tinyllama", "heter"): 3,
    ("dolly", "llama", "homo"): 3,
    ("dolly", "llama", "heter"): 3,
    ("dolly_stratified", "tinyllama", "homo"): 3,
    ("dolly_stratified", "tinyllama", "heter"): 3,
    ("dolly_stratified", "llama", "homo"): 3,
    ("dolly_stratified", "llama", "heter"): 3,
}


def _normalize_accuracy(value: float) -> float:
    return value * 100 if value <= 1 else value


def _read_scores(path: Path) -> list[float]:
    scores = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        value = float(line)
        scores.append(_normalize_accuracy(value))
    return scores


def _score_path(run_dir: Path, seed: int, method: str) -> Path | None:
    seed_dir = run_dir / f"seed{seed}"
    candidates = (
        [seed_dir / "10log.txt", seed_dir / "10" / "log.txt"]
        if method == "flora"
        else [seed_dir / "10" / "log.txt", seed_dir / "10log.txt"]
    )
    return next((path for path in candidates if path.exists()), None)


def _parse_seed(seed_dir: Path) -> int | None:
    if not seed_dir.name.startswith("seed"):
        return None
    seed_text = seed_dir.name.removeprefix("seed")
    return int(seed_text) if seed_text.isdigit() else None


def _empty_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Method key",
            "Method",
            "Dataset",
            "Dataset label",
            "Model key",
            "Model",
            "Setting key",
            "Setting",
            "Local epochs",
            "Config rounds",
            "Seed",
            "Round",
            "Accuracy",
            "Run dir",
            "Score path",
            "Observed rounds",
            "Complete run",
            "Result source",
            "Run status",
        ]
    )


def _iter_base_paths(base_dir: str | Path | Iterable[str | Path]):
    if isinstance(base_dir, str | Path):
        yield Path(base_dir)
        return

    for path in base_dir:
        yield Path(path)


def _iter_tuning_run_dirs(base_dir: str | Path | Iterable[str | Path]):
    search_roots = []
    for base_path in _iter_base_paths(base_dir):
        search_roots.append(base_path)
        grouped_root = base_path / "epoch_round_tuning"
        if grouped_root.is_dir():
            search_roots.append(grouped_root)

    seen = set()
    for root in search_roots:
        for run_dir in sorted(root.glob("tuning-*")):
            if not run_dir.is_dir():
                continue
            key = run_dir.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield run_dir


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _iter_live_log_paths(
    log_dir: str | Path | Iterable[str | Path] = "logs",
    pattern: str = "epoch_round_tuning_*.out",
):
    raw_paths = [log_dir] if isinstance(log_dir, str | Path) else list(log_dir)
    seen = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        path_text = str(raw_path)
        if any(marker in path_text for marker in "*?[]"):
            candidates = path.parent.glob(path.name)
        elif path.is_dir():
            candidates = path.glob(pattern)
        elif path.exists():
            candidates = [path]
        else:
            candidates = []

        for candidate in sorted(candidates):
            key = candidate.resolve(strict=False)
            if key in seen or not candidate.is_file():
                continue
            seen.add(key)
            yield candidate


def _iter_exp2_nonlinear_flora_dirs(base_dir: str | Path | Iterable[str | Path]):
    search_roots = []
    for base_path in _iter_base_paths(base_dir):
        search_roots.append(base_path)
        if not base_path.name.startswith("exp2-"):
            search_roots.extend(path for path in base_path.glob("exp2-*nonlinear*") if path.is_dir())

    seen = set()
    for run_dir in search_roots:
        if not run_dir.is_dir() or not run_dir.name.startswith("exp2-"):
            continue
        key = run_dir.resolve()
        if key in seen:
            continue
        seen.add(key)
        yield run_dir


def _append_run_record(
    run_records: list[dict],
    *,
    method: str,
    dataset: str,
    model: str,
    setting: str,
    epochs: int,
    rounds: int,
    seed: int,
    scores: list[float],
    run_dir: Path,
    score_path: Path,
    observed_rounds: int | None = None,
    complete_run: bool | None = None,
    result_source: str = "score log",
    run_status: str | None = None,
    extra_fields: dict | None = None,
) -> None:
    if not scores:
        return
    observed_rounds = len(scores) if observed_rounds is None else int(observed_rounds)
    complete_run = observed_rounds >= int(rounds) if complete_run is None else bool(complete_run)
    run_status = run_status or ("Complete" if complete_run else "Partial log")
    record = (
        {
            "Method key": method,
            "Method": METHOD_LABELS[method],
            "Dataset": dataset,
            "Dataset label": DATASET_LABELS.get(dataset, dataset),
            "Model key": model,
            "Model": MODEL_LABELS.get(model, model),
            "Setting key": setting,
            "Setting": SETTING_LABELS.get(setting, setting),
            "Local epochs": int(epochs),
            "Config rounds": int(rounds),
            "Seed": seed,
            "Scores": scores,
            "Run dir": str(run_dir),
            "Score path": str(score_path),
            "Observed rounds": observed_rounds,
            "Complete run": complete_run,
            "Result source": result_source,
            "Run status": run_status,
            "Score count": observed_rounds,
        }
    )
    if extra_fields:
        record.update(extra_fields)
    run_records.append(record)


def _records_to_scores_frame(run_records: list[dict]) -> pd.DataFrame:
    if not run_records:
        return _empty_scores_frame()

    row_keys = [
        "Method key",
        "Method",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
        "Local epochs",
        "Config rounds",
        "Seed",
        "Run dir",
        "Score path",
        "Observed rounds",
        "Complete run",
        "Result source",
        "Run status",
    ]
    rows = []
    for record in run_records:
        for round_idx, accuracy in enumerate(record["Scores"], start=1):
            rows.append(
                {
                    key: record[key]
                    for key in row_keys
                    if key in record
                }
                | {
                    "Round": round_idx,
                    "Accuracy": float(accuracy),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["Dataset", "Method key", "Model key", "Setting key", "Local epochs", "Seed", "Round"]
    )


def load_tuning_results(
    base_dir: str | Path | Iterable[str | Path] = ".",
    complete_only: bool = False,
) -> pd.DataFrame:
    """Discover tuning run directories and return one row per evaluated round.

    Pass the repo root, a grouped run directory, legacy exp2 nonlinear FLoRA
    directories, or a list of run directories. Set complete_only=True to skip
    runs whose score log has fewer rows than the configured round count.
    """

    run_records = []

    for run_dir in _iter_tuning_run_dirs(base_dir):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue

        info = match.groupdict()
        method = info["method"]
        model = "llama" if info["model"] == "llama-7b" else info["model"]

        for seed_dir in sorted(run_dir.glob("seed*")):
            seed = _parse_seed(seed_dir)
            if seed is None:
                continue
            score_path = _score_path(run_dir, seed, method)
            if score_path is None:
                continue
            rounds = int(info["rounds"])
            raw_scores = _read_scores(score_path)
            if complete_only and len(raw_scores) < rounds:
                continue
            scores = raw_scores[:rounds]
            _append_run_record(
                run_records,
                method=method,
                dataset=info["dataset"],
                model=model,
                setting=info["setting"],
                epochs=int(info["epochs"]),
                rounds=rounds,
                seed=seed,
                scores=scores,
                run_dir=run_dir,
                score_path=score_path,
                observed_rounds=len(raw_scores),
            )

    for run_dir in _iter_exp2_nonlinear_flora_dirs(base_dir):
        match = EXP2_NONLINEAR_FLORA_DIR_RE.match(run_dir.name)
        if not match:
            continue

        info = match.groupdict()
        method = "nonlinear_flora"
        model = "llama" if info["model"] == "llama" else info["model"]
        setting = info["setting"] or "homo"
        rounds = int(info["rounds"] or (1 if model == "llama" else 3))

        for seed_dir in sorted(run_dir.glob("seed*")):
            seed = _parse_seed(seed_dir)
            if seed is None:
                continue
            score_path = seed_dir / "10" / "log.txt"
            if not score_path.exists():
                continue
            raw_scores = _read_scores(score_path)
            if complete_only and len(raw_scores) < rounds:
                continue
            scores = raw_scores[:rounds]
            _append_run_record(
                run_records,
                method=method,
                dataset=info["dataset"],
                model=model,
                setting=setting,
                epochs=1,
                rounds=rounds,
                seed=seed,
                scores=scores,
                run_dir=run_dir,
                score_path=score_path,
                observed_rounds=len(raw_scores),
            )

    if not run_records:
        return _empty_scores_frame()

    # If both r6 and r10 reruns exist for the same seed/config, use the run with
    # more observed rounds. This avoids double-counting rounds 1..6.
    selected_runs = []
    id_columns = [
        "Method key",
        "Dataset",
        "Model key",
        "Setting key",
        "Local epochs",
        "Seed",
    ]
    for _, group in pd.DataFrame(run_records).groupby(id_columns, sort=False):
        best = group.sort_values(["Score count", "Config rounds"], ascending=False).iloc[0]
        selected_runs.append(best.to_dict())

    return _records_to_scores_frame(selected_runs)


def load_live_tuning_results(
    log_dir: str | Path | Iterable[str | Path] = "logs",
    *,
    run_roots: str | Path | Iterable[str | Path] | None = None,
    pattern: str = "epoch_round_tuning_*.out",
) -> pd.DataFrame:
    """Parse in-progress tuning scores from Slurm stdout logs.

    This is intended for live plotting only. It reads each stdout log's
    output_dir=... line, parses Acc round N: ... lines, and marks the rows as
    incomplete because final log.txt files are still the source of truth for
    complete runs.
    """

    allowed_roots = (
        None
        if run_roots is None
        else [_resolve_path(path) for path in _iter_base_paths(run_roots)]
    )
    run_records = []

    for log_path in _iter_live_log_paths(log_dir, pattern):
        text = log_path.read_text(errors="replace")
        output_matches = list(LIVE_OUTPUT_DIR_RE.finditer(text))
        if not output_matches:
            continue

        output_dir = output_matches[-1].group("output_dir").strip().strip('"').strip("'")
        seed_dir = Path(output_dir.rstrip("/"))
        seed = _parse_seed(seed_dir)
        if seed is None:
            continue

        run_dir = seed_dir.parent
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue

        resolved_run_dir = _resolve_path(run_dir)
        if allowed_roots and not any(_is_relative_to(resolved_run_dir, root) for root in allowed_roots):
            continue

        info = match.groupdict()
        rounds = int(info["rounds"])
        scores_by_round = {}
        for score_match in LIVE_ACCURACY_RE.finditer(text):
            round_idx = int(score_match.group("round"))
            if round_idx >= rounds:
                continue
            scores_by_round[round_idx] = _normalize_accuracy(float(score_match.group("accuracy")))
        if not scores_by_round:
            continue

        scores = [scores_by_round[round_idx] for round_idx in sorted(scores_by_round)]
        method = info["method"]
        model = "llama" if info["model"] == "llama-7b" else info["model"]
        _append_run_record(
            run_records,
            method=method,
            dataset=info["dataset"],
            model=model,
            setting=info["setting"],
            epochs=int(info["epochs"]),
            rounds=rounds,
            seed=seed,
            scores=scores,
            run_dir=run_dir,
            score_path=log_path,
            observed_rounds=len(scores),
            complete_run=False,
            result_source="slurm stdout",
            run_status="Live partial",
            extra_fields={"Log mtime": log_path.stat().st_mtime},
        )

    if not run_records:
        return _empty_scores_frame()

    selected_runs = []
    id_columns = [
        "Method key",
        "Dataset",
        "Model key",
        "Setting key",
        "Local epochs",
        "Seed",
        "Run dir",
    ]
    for _, group in pd.DataFrame(run_records).groupby(id_columns, sort=False):
        best = group.sort_values(["Score count", "Log mtime"], ascending=False).iloc[0]
        selected_runs.append(best.to_dict())

    return _records_to_scores_frame(selected_runs)


def summarize_tuning_results(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    group_columns = [
        "Method key",
        "Method",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
        "Local epochs",
        "Round",
    ]
    for optional_column in ["Result source", "Run status"]:
        if optional_column in scores.columns:
            group_columns.append(optional_column)

    summary = (
        scores.groupby(group_columns)
        .agg(
            **{
                "Mean accuracy": ("Accuracy", "mean"),
                "Std accuracy": ("Accuracy", "std"),
                "Seed count": ("Seed", "nunique"),
                "Seeds": ("Seed", lambda values: ", ".join(str(v) for v in sorted(set(values)))),
                "Max config rounds": ("Config rounds", "max"),
            }
        )
        .reset_index()
    )
    summary["Std accuracy"] = summary["Std accuracy"].fillna(0.0)
    summary["Compute cost"] = summary["Local epochs"] * summary["Round"]
    return summary.sort_values(
        ["Dataset", "Method key", "Model key", "Setting key", "Local epochs", "Round"]
    )


def _future_gain(curve: pd.DataFrame, round_value: int) -> float:
    current = curve.loc[curve["Round"] == round_value, "Mean accuracy"].iloc[0]
    future = curve[(curve["Round"] > round_value) & (curve["Round"] <= round_value + 2)]
    if future.empty:
        return 0.0
    return float(future["Mean accuracy"].max() - current)


def select_plateaus(
    summary: pd.DataFrame,
    tolerance: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-epoch plateau rows and one selected pair per method/dataset/model/setting."""

    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    epoch_group_columns = [
        "Method key",
        "Method",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
        "Local epochs",
    ]
    case_group_columns = [
        "Method key",
        "Method",
        "Dataset",
        "Dataset label",
        "Model key",
        "Model",
        "Setting key",
        "Setting",
    ]

    plateau_rows = []
    for group_values, group in summary.groupby(epoch_group_columns, sort=False):
        curve = group.sort_values("Round")
        best_idx = curve["Mean accuracy"].idxmax()
        best_row = curve.loc[best_idx]
        best_accuracy = float(best_row["Mean accuracy"])
        plateau_row = None

        for _, row in curve.iterrows():
            within_best = best_accuracy - float(row["Mean accuracy"]) <= tolerance
            no_large_future_gain = _future_gain(curve, int(row["Round"])) <= tolerance
            if within_best and no_large_future_gain:
                plateau_row = row
                break

        if plateau_row is None:
            plateau_row = best_row

        base = dict(zip(epoch_group_columns, group_values))
        later_rounds = curve[curve["Round"] > int(best_row["Round"])]
        declined_after_best = bool(
            not later_rounds.empty
            and later_rounds["Mean accuracy"].min() < best_accuracy - tolerance
        )
        plateau_rows.append(
            base
            | {
                "Plateau round": int(plateau_row["Round"]),
                "Plateau accuracy": float(plateau_row["Mean accuracy"]),
                "Best round": int(best_row["Round"]),
                "Best accuracy": best_accuracy,
                "Max round observed": int(curve["Round"].max()),
                "Declined after best": declined_after_best,
            }
        )

    selected_rows = []
    for group_values, group in summary.groupby(case_group_columns, sort=False):
        best_idx = group["Mean accuracy"].idxmax()
        best_row = group.loc[best_idx]
        best_accuracy = float(best_row["Mean accuracy"])
        eligible = group[group["Mean accuracy"] >= best_accuracy - tolerance].copy()
        eligible = eligible.sort_values(
            ["Compute cost", "Round", "Local epochs", "Mean accuracy"],
            ascending=[True, True, True, False],
        )
        selected = eligible.iloc[0]
        selected_cost = int(selected["Compute cost"])
        same_low_cost = eligible[eligible["Compute cost"] == selected_cost]

        base = dict(zip(case_group_columns, group_values))
        selected_rows.append(
            base
            | {
                "Selected epochs": int(selected["Local epochs"]),
                "Selected round": int(selected["Round"]),
                "Selected accuracy": float(selected["Mean accuracy"]),
                "Selected std": float(selected["Std accuracy"]),
                "Selected seed count": int(selected["Seed count"]),
                "Selected seeds": selected["Seeds"],
                "Selected compute cost": selected_cost,
                "Best epochs": int(best_row["Local epochs"]),
                "Best round": int(best_row["Round"]),
                "Best accuracy": best_accuracy,
                "Accuracy gap to best": best_accuracy - float(selected["Mean accuracy"]),
                "Low-cost tie count": int(len(same_low_cost)),
            }
        )

    return (
        pd.DataFrame(plateau_rows).sort_values(case_group_columns + ["Local epochs"]),
        pd.DataFrame(selected_rows).sort_values(case_group_columns),
    )


def build_extension_requests(
    summary: pd.DataFrame,
    source_rounds: int = 6,
    target_rounds: int = 10,
    improvement_threshold: float = 0.5,
) -> pd.DataFrame:
    """Find curves whose last two one-round deltas both exceed the threshold."""

    if summary.empty:
        return _empty_manifest_frame()

    request_rows = []
    group_columns = ["Method key", "Dataset", "Model key", "Setting key", "Local epochs"]
    for group_values, group in summary.groupby(group_columns, sort=False):
        curve = group.sort_values("Round")
        if int(curve["Round"].max()) != source_rounds:
            continue
        best_round = int(curve.loc[curve["Mean accuracy"].idxmax(), "Round"])
        if best_round != source_rounds:
            continue
        if len(curve) < 3:
            continue

        last_three = curve.tail(3)
        values = last_three["Mean accuracy"].to_numpy()
        delta_prev = float(values[1] - values[0])
        delta_last = float(values[2] - values[1])
        if delta_prev <= improvement_threshold or delta_last <= improvement_threshold:
            continue

        method, dataset, model, setting, epochs = group_values
        seeds = _parse_seed_list(curve.iloc[-1]["Seeds"])
        for seed in seeds:
            request_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": model,
                    "setting": setting,
                    "epochs": int(epochs),
                    "rounds": int(target_rounds),
                    "seed": int(seed),
                    "delta_prev": delta_prev,
                    "delta_last": delta_last,
                }
            )

    if not request_rows:
        return _empty_manifest_frame()
    return pd.DataFrame(request_rows)


def build_repeat_requests(
    summary: pd.DataFrame,
    model: str = "tinyllama",
    top_n_epochs: int = 2,
    repeat_seeds: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Select the best epoch settings per case and request repeat seeds."""

    if summary.empty:
        return _empty_manifest_frame()

    request_rows = []
    filtered = summary[summary["Model key"] == model]
    group_columns = ["Method key", "Dataset", "Model key", "Setting key"]

    for group_values, group in filtered.groupby(group_columns, sort=False):
        epoch_scores = (
            group.groupby("Local epochs")
            .agg(
                best_accuracy=("Mean accuracy", "max"),
                rounds=("Round", "max"),
                compute_at_best=("Compute cost", "min"),
            )
            .reset_index()
            .sort_values(["best_accuracy", "compute_at_best"], ascending=[False, True])
            .head(top_n_epochs)
        )
        method, dataset, model_key, setting = group_values
        for _, epoch_row in epoch_scores.iterrows():
            for seed in repeat_seeds:
                request_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "model": model_key,
                        "setting": setting,
                        "epochs": int(epoch_row["Local epochs"]),
                        "rounds": int(epoch_row["rounds"]),
                        "seed": int(seed),
                    }
                )

    if not request_rows:
        return _empty_manifest_frame()
    return pd.DataFrame(request_rows).drop_duplicates(MANIFEST_COLUMNS)


def build_llama_confirmation_requests(
    selected_pairs: pd.DataFrame,
    seed: int = 0,
    rounds: int = 3,
) -> pd.DataFrame:
    """Create Llama-7B confirmation rows from selected TinyLlama epochs."""

    if selected_pairs.empty:
        return _empty_manifest_frame()

    request_rows = []
    tiny_selected = selected_pairs[selected_pairs["Model key"] == "tinyllama"]
    for _, row in tiny_selected.iterrows():
        epochs_to_run = sorted({1, int(row["Selected epochs"])})
        for epochs in epochs_to_run:
            request_rows.append(
                {
                    "method": row["Method key"],
                    "dataset": row["Dataset"],
                    "model": "llama",
                    "setting": row["Setting key"],
                    "epochs": int(epochs),
                    "rounds": int(rounds),
                    "seed": int(seed),
                }
                )

    if not request_rows:
        return _empty_manifest_frame()
    return pd.DataFrame(request_rows).drop_duplicates(MANIFEST_COLUMNS)


def write_manifest(requests: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in manifest_records(requests):
            writer.writerow(row)


def manifest_records(requests: pd.DataFrame) -> list[dict[str, int | str]]:
    if requests.empty:
        return []
    missing = [column for column in MANIFEST_COLUMNS if column not in requests.columns]
    if missing:
        raise ValueError(f"Request frame is missing manifest columns: {missing}")
    return requests[MANIFEST_COLUMNS].drop_duplicates().to_dict("records")


def make_tuning_round_curves(
    summary: pd.DataFrame,
    dataset: str,
    model: str,
    setting: str | None = None,
    include_paper_baseline: bool = False,
):
    data = summary[(summary["Dataset"] == dataset) & (summary["Model key"] == model)]
    if setting is not None:
        data = data[data["Setting key"] == setting]
    if data.empty:
        return None

    plot_data = data.copy()
    plot_data["Local epochs"] = plot_data["Local epochs"].astype(str)

    if include_paper_baseline:
        baseline_rows = []
        for setting_key in sorted(data["Setting key"].unique()):
            baseline = PAPER_BASELINES.get((dataset, model, setting_key))
            reported_round = PAPER_BASELINE_ROUNDS.get((dataset, model, setting_key))
            if baseline is None or reported_round is None:
                continue
            setting_label = SETTING_LABELS.get(setting_key, setting_key)
            baseline_rows.append(
                {
                    **{column: None for column in plot_data.columns},
                    "Method": "FLoRA paper reported point",
                    "Setting": setting_label,
                    "Setting key": setting_key,
                    "Local epochs": f"paper r{reported_round}",
                    "Round": reported_round,
                    "Mean accuracy": baseline,
                }
            )
        if baseline_rows:
            plot_data = pd.concat([plot_data, pd.DataFrame(baseline_rows)], ignore_index=True)

    fig = px.line(
        plot_data,
        x="Round",
        y="Mean accuracy",
        color="Local epochs",
        line_dash="Method",
        facet_col="Setting" if setting is None else None,
        markers=True,
        template="plotly_white",
        title=f"{DATASET_LABELS.get(dataset, dataset)} {MODEL_LABELS.get(model, model)} tuning curves",
        labels={"Mean accuracy": "MMLU accuracy (%)", "Local epochs": "Local epochs"},
    )
    fig.update_layout(height=480, legend_title_text="")
    fig.update_xaxes(dtick=1)
    return fig


def compare_selected_to_paper(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()

    rows = []
    for _, row in selected_pairs.iterrows():
        baseline = PAPER_BASELINES.get((row["Dataset"], row["Model key"], row["Setting key"]))
        if baseline is None:
            continue
        rows.append(
            {
                "Method": row["Method"],
                "Dataset": row["Dataset label"],
                "Model": row["Model"],
                "Setting": row["Setting"],
                "Selected epochs": int(row["Selected epochs"]),
                "Selected round": int(row["Selected round"]),
                "Selected accuracy": float(row["Selected accuracy"]),
                "Best epochs": int(row["Best epochs"]),
                "Best round": int(row["Best round"]),
                "Best accuracy": float(row["Best accuracy"]),
                "FLoRA paper": float(baseline),
                "FLoRA paper round": int(PAPER_BASELINE_ROUNDS[(row["Dataset"], row["Model key"], row["Setting key"])]),
                "Selected delta vs paper": float(row["Selected accuracy"]) - float(baseline),
                "Best delta vs paper": float(row["Best accuracy"]) - float(baseline),
            }
        )

    return pd.DataFrame(rows)


def make_tuning_heatmap(
    summary: pd.DataFrame,
    method: str,
    dataset: str,
    model: str,
    setting: str,
):
    data = summary[
        (summary["Method key"] == method)
        & (summary["Dataset"] == dataset)
        & (summary["Model key"] == model)
        & (summary["Setting key"] == setting)
    ]
    if data.empty:
        return None

    heatmap = data.pivot_table(
        index="Local epochs",
        columns="Round",
        values="Mean accuracy",
        aggfunc="mean",
    ).sort_index()
    title = (
        f"{METHOD_LABELS.get(method, method)} "
        f"{DATASET_LABELS.get(dataset, dataset)} "
        f"{MODEL_LABELS.get(model, model)} {SETTING_LABELS.get(setting, setting)}"
    )
    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_white",
        title=f"{title}: epochs x rounds",
        labels={"x": "Round", "y": "Local epochs", "color": "MMLU accuracy (%)"},
    )
    fig.update_layout(height=420)
    return fig


def _empty_manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _parse_seed_list(seed_text: str) -> list[int]:
    seeds = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds
