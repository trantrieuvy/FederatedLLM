#!/usr/bin/env python3
"""Generate Plotly diagnostics from federated local_metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px


METRICS_FILENAME = "local_metrics.jsonl"
REQUIRED_COLUMNS = {
    "method",
    "round",
    "client_id",
    "local_epoch",
    "train_loss",
    "eval_loss",
}


def discover_metric_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob(METRICS_FILENAME)))
        else:
            raise FileNotFoundError(f"Could not find metrics path: {path}")
    unique_files = list(dict.fromkeys(file.resolve() for file in files))
    if not unique_files:
        raise FileNotFoundError(f"No {METRICS_FILENAME} files found.")
    return unique_files


def load_local_metrics(metric_files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in metric_files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                row["Run path"] = str(path.parent)
                rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("Metric files contain no records.")
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Metric records are missing columns: {missing}")
    return data


def summarize_local_metrics(data: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["Run path", "method", "round", "local_epoch", "train_on_inputs"]
    summary = (
        data.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            train_loss=("train_loss", "mean"),
            eval_loss=("eval_loss", "mean"),
            train_perplexity=("train_perplexity", "mean"),
            eval_perplexity=("eval_perplexity", "mean"),
            client_count=("client_id", "nunique"),
        )
        .sort_values(["Run path", "method", "round", "local_epoch"])
    )
    summary["generalization_gap"] = summary["eval_loss"] - summary["train_loss"]
    return summary


def write_plots(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_local_metrics(data)
    summary_path = output_dir / "local_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    loss_data = summary.melt(
        id_vars=["Run path", "method", "round", "local_epoch"],
        value_vars=["train_loss", "eval_loss"],
        var_name="Split",
        value_name="Loss",
    )
    loss_data["Split"] = loss_data["Split"].map(
        {"train_loss": "Monitored train sample", "eval_loss": "Local validation"}
    )
    loss_figure = px.line(
        loss_data,
        x="local_epoch",
        y="Loss",
        color="Split",
        line_dash="method",
        facet_col="round",
        facet_col_wrap=5,
        markers=True,
        hover_data=["Run path"],
        labels={"local_epoch": "Local epoch", "round": "Communication round"},
        title="Mean Local Training And Validation Loss",
    )
    loss_path = output_dir / "mean_local_loss.html"
    loss_figure.write_html(loss_path)

    gap_figure = px.line(
        summary,
        x="local_epoch",
        y="generalization_gap",
        color="method",
        line_dash="Run path",
        facet_col="round",
        facet_col_wrap=5,
        markers=True,
        labels={
            "local_epoch": "Local epoch",
            "generalization_gap": "Validation loss - train loss",
            "round": "Communication round",
        },
        title="Mean Local Generalization Gap",
    )
    gap_path = output_dir / "mean_local_generalization_gap.html"
    gap_figure.write_html(gap_path)

    client_data = data.copy()
    client_data["Client"] = client_data["client_id"].astype(str)
    client_figure = px.line(
        client_data,
        x="local_epoch",
        y="eval_loss",
        color="Client",
        line_dash="method",
        facet_col="round",
        facet_col_wrap=5,
        markers=True,
        hover_data=["Run path"],
        labels={"local_epoch": "Local epoch", "eval_loss": "Validation loss"},
        title="Per-Client Local Validation Loss",
    )
    client_path = output_dir / "per_client_validation_loss.html"
    client_figure.write_html(client_path)
    return [summary_path, loss_path, gap_path, client_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        type=Path,
        help="JSONL files or run/root directories containing local_metrics.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_local_monitoring"),
        help="Directory in which Plotly HTML plots and a summary CSV are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_files = discover_metric_files(args.metrics_paths)
    data = load_local_metrics(metric_files)
    outputs = write_plots(data, args.output_dir)
    print(f"Loaded {len(data)} metric rows from {len(metric_files)} files.")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
