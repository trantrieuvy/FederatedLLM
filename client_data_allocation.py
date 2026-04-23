import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ALPHA = 0.5
DEFAULT_SEED = 42
DEFAULT_TEST_PER_CATEGORY = 10
MIN_DIRICHLET_CLIENT_SIZE = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate federated Dolly dataset splits."
    )
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("dirichlet", "stratified_keep_sizes"),
        required=True,
        help="dirichlet recreates the legacy non-IID split; "
        "stratified_keep_sizes rebalances categories while preserving legacy client sizes.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Dirichlet concentration parameter for non-IID partitioning.",
    )
    parser.add_argument(
        "--output-root",
        default="data_dolly",
        help="Dataset root to write, e.g. data_dolly or data_dolly_stratified.",
    )
    parser.add_argument(
        "--source-root",
        default="data_dolly",
        help="Existing Dolly split root to reuse for preserved global train/test and client sizes.",
    )
    parser.add_argument(
        "--dataset-path",
        default="new-databricks-dolly-15k.json",
        help="Raw Dolly JSONL file used for legacy Dirichlet generation.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--test-per-category",
        type=int,
        default=DEFAULT_TEST_PER_CATEGORY,
        help="Balanced holdout size per Dolly category when creating a fresh legacy split.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as infile:
        return json.load(infile)


def write_json(path, payload):
    with open(path, "w") as outfile:
        json.dump(payload, outfile)


def make_output_dir(root, num_clients):
    output_dir = Path(root) / str(num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def category_counter(records):
    return {key: int(value) for key, value in sorted(Counter(r["category"] for r in records).items())}


def load_raw_dolly_split(dataset_path, test_per_category):
    df = pd.read_json(dataset_path, orient="records", lines=True)
    sorted_df = df.sort_values(by=["category"])
    grouped = sorted_df.groupby("category")
    sampled_df = grouped.apply(lambda group: group.sample(n=test_per_category))
    sampled_df = sampled_df.reset_index(level=0, drop=True)
    remaining_df = sorted_df.drop(index=sampled_df.index)

    sampled_df = sampled_df.reset_index().drop("index", axis=1)
    remaining_df = remaining_df.reset_index().drop("index", axis=1)
    return remaining_df, sampled_df


def dirichlet_partition_indices(remaining_df, num_clients, alpha):
    min_size = 0
    num_rows = len(remaining_df)
    category_names = remaining_df["category"].unique().tolist()

    while min_size < MIN_DIRICHLET_CLIENT_SIZE:
        idx_partition = [[] for _ in range(num_clients)]
        for category_name in category_names:
            category_rows = remaining_df.loc[remaining_df["category"] == category_name]
            category_row_indices = category_rows.index.values
            np.random.shuffle(category_row_indices)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    proportion * (len(client_indices) < num_rows / num_clients)
                    for proportion, client_indices in zip(proportions, idx_partition)
                ]
            )
            proportions = proportions / proportions.sum()
            split_points = (
                np.cumsum(proportions) * len(category_row_indices)
            ).astype(int)[:-1]
            idx_partition = [
                client_indices + split_indices.tolist()
                for client_indices, split_indices in zip(
                    idx_partition, np.split(category_row_indices, split_points)
                )
            ]
            min_size = min(len(client_indices) for client_indices in idx_partition)

    return idx_partition


def save_split_records(output_dir, global_training_records, global_test_records, client_records, metadata):
    write_json(output_dir / "global_training.json", global_training_records)
    write_json(output_dir / "global_test.json", global_test_records)

    for client_id, records in enumerate(client_records):
        write_json(output_dir / f"local_training_{client_id}.json", records)

    write_json(output_dir / "split_metadata.json", metadata)


def allocate_category_counts(total, capacities):
    if total > sum(capacities):
        raise ValueError("Category total exceeds remaining client capacities.")
    if total == 0:
        return [0 for _ in capacities]

    remaining_capacity = sum(capacities)
    raw_shares = [(total * capacity) / remaining_capacity for capacity in capacities]
    allocation = [int(np.floor(raw_share)) for raw_share in raw_shares]
    remainder = total - sum(allocation)

    fractional_order = sorted(
        range(len(capacities)),
        key=lambda client_id: (
            -(raw_shares[client_id] - allocation[client_id]),
            -capacities[client_id],
            client_id,
        ),
    )

    while remainder > 0:
        assigned = False
        for client_id in fractional_order:
            if allocation[client_id] < capacities[client_id]:
                allocation[client_id] += 1
                remainder -= 1
                assigned = True
                if remainder == 0:
                    break
        if not assigned:
            raise ValueError("Unable to finish largest-remainder allocation within capacity bounds.")

    return allocation


def build_stratified_clients(global_training_records, client_quotas, seed):
    rng = random.Random(seed)
    records_by_category = defaultdict(list)
    for record in global_training_records:
        records_by_category[record["category"]].append(record)

    categories = sorted(
        records_by_category,
        key=lambda category_name: (-len(records_by_category[category_name]), category_name),
    )
    for category_name in categories:
        rng.shuffle(records_by_category[category_name])

    remaining_capacities = list(client_quotas)
    client_records = [[] for _ in client_quotas]
    client_category_counts = {client_id: {} for client_id in range(len(client_quotas))}

    for category_index, category_name in enumerate(categories):
        category_records = records_by_category[category_name]
        if category_index == len(categories) - 1:
            allocation = list(remaining_capacities)
        else:
            allocation = allocate_category_counts(len(category_records), remaining_capacities)

        offset = 0
        for client_id, category_count in enumerate(allocation):
            if category_count == 0:
                continue
            next_offset = offset + category_count
            client_records[client_id].extend(category_records[offset:next_offset])
            client_category_counts[client_id][category_name] = int(category_count)
            remaining_capacities[client_id] -= category_count
            offset = next_offset

        if offset != len(category_records):
            raise ValueError(f"Unassigned records remain for category {category_name}.")

    if any(remaining_capacities):
        raise ValueError("Stratified allocation did not exhaust all client capacities.")

    return client_records, client_category_counts


def metadata_for_dirichlet(args, global_training_records, global_test_records, client_records):
    return {
        "split_mode": args.mode,
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "num_clients": int(args.num_clients),
        "dataset_path": args.dataset_path,
        "output_root": args.output_root,
        "source_root": None,
        "global_test_policy": f"balanced_holdout_{args.test_per_category}_per_category_from_raw_dataset",
        "client_size_policy": f"dirichlet_alpha_{args.alpha}_min_client_size_{MIN_DIRICHLET_CLIENT_SIZE}",
        "global_training_size": int(len(global_training_records)),
        "global_test_size": int(len(global_test_records)),
        "client_sizes": [int(len(records)) for records in client_records],
        "global_training_category_counts": category_counter(global_training_records),
        "global_test_category_counts": category_counter(global_test_records),
        "client_category_counts": {
            str(client_id): category_counter(records)
            for client_id, records in enumerate(client_records)
        },
    }


def metadata_for_stratified(args, global_training_records, global_test_records, client_records, client_category_counts):
    return {
        "split_mode": args.mode,
        "seed": int(args.seed),
        "alpha": None,
        "num_clients": int(args.num_clients),
        "dataset_path": None,
        "output_root": args.output_root,
        "source_root": args.source_root,
        "global_test_policy": "copied_unchanged_from_source_root",
        "client_size_policy": "preserve_legacy_client_sizes_from_source_root",
        "global_training_size": int(len(global_training_records)),
        "global_test_size": int(len(global_test_records)),
        "client_sizes": [int(len(records)) for records in client_records],
        "global_training_category_counts": category_counter(global_training_records),
        "global_test_category_counts": category_counter(global_test_records),
        "client_category_counts": {
            str(client_id): {
                category_name: int(count)
                for category_name, count in sorted(client_category_counts[client_id].items())
            }
            for client_id in range(len(client_records))
        },
    }


def create_legacy_dirichlet_split(args):
    remaining_df, sampled_df = load_raw_dolly_split(args.dataset_path, args.test_per_category)
    idx_partition = dirichlet_partition_indices(remaining_df, args.num_clients, args.alpha)

    output_dir = make_output_dir(args.output_root, args.num_clients)
    global_training_records = remaining_df.to_dict(orient="records")
    global_test_records = sampled_df.to_dict(orient="records")

    client_records = []
    for idx in idx_partition:
        client_df = remaining_df.loc[idx]
        client_df = client_df.reset_index().drop("index", axis=1)
        client_records.append(client_df.to_dict(orient="records"))

    metadata = metadata_for_dirichlet(
        args=args,
        global_training_records=global_training_records,
        global_test_records=global_test_records,
        client_records=client_records,
    )
    save_split_records(output_dir, global_training_records, global_test_records, client_records, metadata)


def create_stratified_split(args):
    source_dir = Path(args.source_root) / str(args.num_clients)
    output_dir = make_output_dir(args.output_root, args.num_clients)

    if source_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "For stratified_keep_sizes, output-root must differ from source-root so the legacy split stays untouched."
        )

    source_global_training_path = source_dir / "global_training.json"
    source_global_test_path = source_dir / "global_test.json"
    if not source_global_training_path.exists() or not source_global_test_path.exists():
        raise FileNotFoundError(
            f"Could not find legacy Dolly split in {source_dir}. Expected global_training.json and global_test.json."
        )

    global_training_records = load_json(source_global_training_path)
    global_test_records = load_json(source_global_test_path)

    client_quotas = []
    for client_id in range(args.num_clients):
        client_path = source_dir / f"local_training_{client_id}.json"
        if not client_path.exists():
            raise FileNotFoundError(f"Could not find legacy client split {client_path}.")
        client_quotas.append(len(load_json(client_path)))

    client_records, client_category_counts = build_stratified_clients(
        global_training_records=global_training_records,
        client_quotas=client_quotas,
        seed=args.seed,
    )

    shutil.copy2(source_global_training_path, output_dir / "global_training.json")
    shutil.copy2(source_global_test_path, output_dir / "global_test.json")
    for client_id, records in enumerate(client_records):
        write_json(output_dir / f"local_training_{client_id}.json", records)

    metadata = metadata_for_stratified(
        args=args,
        global_training_records=global_training_records,
        global_test_records=global_test_records,
        client_records=client_records,
        client_category_counts=client_category_counts,
    )
    write_json(output_dir / "split_metadata.json", metadata)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "dirichlet":
        create_legacy_dirichlet_split(args)
    else:
        create_stratified_split(args)

    print(f"Dolly split written to {Path(args.output_root) / str(args.num_clients)}")


if __name__ == "__main__":
    main()
