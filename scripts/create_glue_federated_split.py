#!/usr/bin/env python3
"""Create deterministic federated client splits for GLUE tasks.

For RTE, the expected output is:

  data_rte_stratified/10/local_training_0.json
  ...
  data_rte_stratified/10/local_training_9.json
  data_rte_stratified/10/global_val.json
  data_rte_stratified/10/split_metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="rte", choices=sorted(TASK_TO_KEYS))
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--output-root", default="data_rte_stratified")
    parser.add_argument(
        "--source-split-dir",
        type=Path,
        help="Reuse local_training_*.json and global_val.json from an existing split instead of loading GLUE.",
    )
    parser.add_argument(
        "--mode",
        choices=("stratified", "iid"),
        default="stratified",
        help="stratified preserves label balance as much as possible across clients.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def project_records(records, sentence1_key: str, sentence2_key: str | None) -> list[dict]:
    projected = []
    for record in records:
        label = int(record["label"])
        if label < 0:
            continue
        item = {
            sentence1_key: record[sentence1_key],
            "label": label,
        }
        if sentence2_key is not None:
            item[sentence2_key] = record[sentence2_key]
        projected.append(item)
    return projected


def split_iid(records: list[dict], num_clients: int, rng: random.Random) -> list[list[dict]]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    clients = [[] for _ in range(num_clients)]
    for index, record in enumerate(shuffled):
        clients[index % num_clients].append(record)
    return clients


def split_stratified(records: list[dict], num_clients: int, rng: random.Random) -> list[list[dict]]:
    by_label: dict[int, list[dict]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)

    clients = [[] for _ in range(num_clients)]
    for label in sorted(by_label):
        label_records = by_label[label]
        rng.shuffle(label_records)
        for index, record in enumerate(label_records):
            clients[index % num_clients].append(record)

    for client_records in clients:
        rng.shuffle(client_records)
    return clients


def label_counts(records: list[dict]) -> dict[int, int]:
    return {int(key): int(value) for key, value in sorted(Counter(int(r["label"]) for r in records).items())}


def load_source_records(
    args: argparse.Namespace,
    task_name: str,
    sentence1_key: str,
    sentence2_key: str | None,
) -> tuple[list[dict], list[dict]]:
    if args.source_split_dir is not None:
        source_paths = sorted(
            args.source_split_dir.glob("local_training_*.json"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        if not source_paths:
            raise FileNotFoundError(f"No local training records found in {args.source_split_dir}")
        train_records = []
        for source_path in source_paths:
            train_records.extend(read_json(source_path))
        val_records = read_json(args.source_split_dir / "global_val.json")
        return train_records, val_records

    dataset = load_dataset("glue", task_name)
    train_records = project_records(dataset["train"], sentence1_key, sentence2_key)
    validation_split = "validation_matched" if task_name == "mnli" else "validation"
    val_records = project_records(dataset[validation_split], sentence1_key, sentence2_key)
    return train_records, val_records


def main() -> None:
    args = parse_args()
    task_name = args.task_name.lower()
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]
    rng = random.Random(args.seed)

    train_records, val_records = load_source_records(args, task_name, sentence1_key, sentence2_key)

    if args.mode == "stratified":
        clients = split_stratified(train_records, args.num_clients, rng)
    else:
        clients = split_iid(train_records, args.num_clients, rng)

    output_dir = Path(args.output_root) / str(args.num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)
    for client_id, client_records in enumerate(clients):
        write_json(output_dir / f"local_training_{client_id}.json", client_records)
    write_json(output_dir / "global_val.json", val_records)

    metadata = {
        "task_name": task_name,
        "mode": args.mode,
        "seed": int(args.seed),
        "num_clients": int(args.num_clients),
        "train_size": len(train_records),
        "validation_size": len(val_records),
        "train_label_counts": label_counts(train_records),
        "validation_label_counts": label_counts(val_records),
        "source_split_dir": str(args.source_split_dir) if args.source_split_dir is not None else None,
        "client_sizes": {client_id: len(records) for client_id, records in enumerate(clients)},
        "client_label_counts": {
            client_id: label_counts(records)
            for client_id, records in enumerate(clients)
        },
    }
    write_json(output_dir / "split_metadata.json", metadata)
    print(f"Wrote {len(train_records)} train and {len(val_records)} validation records to {output_dir}")


if __name__ == "__main__":
    main()
