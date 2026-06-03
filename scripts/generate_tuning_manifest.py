#!/usr/bin/env python3
"""Generate tab-separated manifests for epoch/round tuning jobs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path


MANIFEST_COLUMNS = ["method", "dataset", "model", "setting", "epochs", "rounds", "seed"]

METHODS = ["flora", "ffa"]
DATASETS = ["wiz", "dolly_stratified"]
SETTINGS = ["homo", "heter"]
TINY_EPOCHS = [1, 2, 3, 5]


def write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def smoke_rows() -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "dataset": "dolly_stratified",
            "model": "tinyllama",
            "setting": "homo",
            "epochs": 1,
            "rounds": 1,
            "seed": 0,
        }
        for method in METHODS
    ]


def tinyllama_coarse_rows() -> list[dict[str, object]]:
    rows = []
    for method, dataset, setting, epochs in product(
        METHODS,
        DATASETS,
        SETTINGS,
        TINY_EPOCHS,
    ):
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "model": "tinyllama",
                "setting": setting,
                "epochs": epochs,
                "rounds": 6,
                "seed": 0,
            }
        )
    return rows


def custom_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for method, dataset, model, setting, epochs, seed in product(
        args.methods,
        args.datasets,
        args.models,
        args.settings,
        args.epochs,
        args.seeds,
    ):
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "model": model,
                "setting": setting,
                "epochs": epochs,
                "rounds": args.rounds,
                "seed": seed,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["smoke", "tinyllama-coarse", "custom"],
        default="tinyllama-coarse",
    )
    parser.add_argument("--output", type=Path, default=Path("tuning_manifests/tinyllama_coarse.tsv"))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=["tinyllama"])
    parser.add_argument("--settings", nargs="+", default=SETTINGS)
    parser.add_argument("--epochs", nargs="+", type=int, default=TINY_EPOCHS)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "smoke":
        rows = smoke_rows()
    elif args.phase == "tinyllama-coarse":
        rows = tinyllama_coarse_rows()
    else:
        rows = custom_rows(args)

    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
