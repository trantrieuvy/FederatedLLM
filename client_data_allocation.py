import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ALPHA = 0.5
DEFAULT_SEED = 42
DEFAULT_TEST_PER_CATEGORY = 10
MIN_DIRICHLET_CLIENT_SIZE = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate federated dataset splits."
    )
    parser.add_argument(
        "--dataset",
        choices=("dolly", "wizard"),
        default="dolly",
        help="Dataset family to split. Dolly supports dirichlet and stratified modes; "
        "Wizard supports stratified_keep_sizes.",
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
        help="Existing split root to reuse for preserved client sizes.",
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
    parser.add_argument(
        "--num-length-buckets",
        type=int,
        default=5,
        help="Wizard-only: number of instruction-length buckets used inside each task family.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as infile:
        return json.load(infile)


def write_json(path, payload):
    with open(path, "w") as outfile:
        json.dump(payload, outfile)


def require_numpy():
    import numpy as np

    return np


def require_pandas():
    import pandas as pd

    return pd


def make_output_dir(root, num_clients):
    output_dir = Path(root) / str(num_clients)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def category_counter(records):
    return {key: int(value) for key, value in sorted(Counter(r["category"] for r in records).items())}


def label_counter(labels):
    return {key: int(value) for key, value in sorted(Counter(labels).items())}


def wizard_task_family(record):
    text = str(record.get("instruction", "")).lower()

    if any(
        marker in text
        for marker in (
            "python",
            "javascript",
            "java ",
            "c++",
            "sql",
            "html",
            "css",
            "matlab",
            "code",
            "program",
            "function",
            "algorithm",
            "debug",
        )
    ):
        return "code"
    if any(
        marker in text
        for marker in (
            "calculate",
            "equation",
            "solve",
            "probability",
            "statistics",
            "mathematical",
            "formula",
            "area of",
            "volume of",
        )
    ):
        return "math"
    if any(
        marker in text
        for marker in (
            "translate",
            "rewrite",
            "rephrase",
            "paraphrase",
            "grammar",
            "edit",
            "convert",
        )
    ):
        return "rewrite_translate"
    if any(
        marker in text
        for marker in (
            "summarize",
            "summary",
            "extract",
            "classify",
            "categorize",
            "sentiment",
        )
    ):
        return "analysis_classification"
    if any(
        marker in text
        for marker in (
            "story",
            "poem",
            "song",
            "script",
            "dialogue",
            "creative",
            "write a",
        )
    ):
        return "creative_writing"
    if any(
        marker in text
        for marker in (
            "recommend",
            "suggest",
            "advice",
            "tips",
            "best way",
            "healthy",
        )
    ):
        return "recommendation_advice"
    if any(
        marker in text
        for marker in (
            "plan",
            "schedule",
            "itinerary",
            "steps",
            "strategy",
            "process",
        )
    ):
        return "planning"
    if any(
        marker in text
        for marker in (
            "table",
            "json",
            "csv",
            "excel",
            "database",
            "dataset",
            "chart",
            "graph",
        )
    ):
        return "structured_data"
    if text.startswith(("what ", "why ", "how ", "when ", "where ", "who ")):
        return "qa_explanation"
    return "general"


def word_count(text):
    return len(str(text).split())


def quantile_edges(values, num_buckets):
    if num_buckets < 1:
        raise ValueError("--num-length-buckets must be at least 1.")
    if num_buckets == 1:
        return []

    sorted_values = sorted(values)
    quantiles = [index / num_buckets for index in range(1, num_buckets)]
    edges = []
    for quantile in quantiles:
        position = quantile * (len(sorted_values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            edges.append(sorted_values[lower])
        else:
            fraction = position - lower
            interpolated = sorted_values[lower] + (
                sorted_values[upper] - sorted_values[lower]
            ) * fraction
            edges.append(interpolated)
    return sorted({int(edge) for edge in edges})


def bucket_for(value, edges):
    for bucket, edge in enumerate(edges):
        if value <= edge:
            return bucket
    return len(edges)


def wizard_stratification_labels(records, num_length_buckets):
    lengths = [word_count(record.get("instruction", "")) for record in records]
    edges = quantile_edges(lengths, num_length_buckets)
    labels = [
        f"{wizard_task_family(record)}__instruction_len_q{bucket_for(length, edges)}"
        for record, length in zip(records, lengths)
    ]
    return labels, edges


def load_raw_dolly_split(dataset_path, test_per_category):
    pd = require_pandas()
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
    np = require_numpy()
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


def allocate_label_counts(total, capacities):
    if total > sum(capacities):
        raise ValueError("Label total exceeds remaining client capacities.")
    if total == 0:
        return [0 for _ in capacities]

    remaining_capacity = sum(capacities)
    raw_shares = [(total * capacity) / remaining_capacity for capacity in capacities]
    allocation = [int(math.floor(raw_share)) for raw_share in raw_shares]
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


def build_stratified_clients(global_training_records, client_quotas, seed, labels=None):
    rng = random.Random(seed)
    if labels is None:
        labels = [record["category"] for record in global_training_records]

    records_by_label = defaultdict(list)
    for record, label in zip(global_training_records, labels):
        records_by_label[label].append(record)

    label_names = sorted(
        records_by_label,
        key=lambda label_name: (-len(records_by_label[label_name]), label_name),
    )
    for label_name in label_names:
        rng.shuffle(records_by_label[label_name])

    remaining_capacities = list(client_quotas)
    client_records = [[] for _ in client_quotas]
    client_label_counts = {client_id: {} for client_id in range(len(client_quotas))}

    for label_index, label_name in enumerate(label_names):
        label_records = records_by_label[label_name]
        if label_index == len(label_names) - 1:
            allocation = list(remaining_capacities)
        else:
            allocation = allocate_label_counts(len(label_records), remaining_capacities)

        offset = 0
        for client_id, label_count in enumerate(allocation):
            if label_count == 0:
                continue
            next_offset = offset + label_count
            client_records[client_id].extend(label_records[offset:next_offset])
            client_label_counts[client_id][label_name] = int(label_count)
            remaining_capacities[client_id] -= label_count
            offset = next_offset

        if offset != len(label_records):
            raise ValueError(f"Unassigned records remain for label {label_name}.")

    if any(remaining_capacities):
        raise ValueError("Stratified allocation did not exhaust all client capacities.")

    for records in client_records:
        rng.shuffle(records)

    return client_records, client_label_counts


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


def metadata_for_dolly_stratified(args, global_training_records, global_test_records, client_records, client_category_counts):
    return {
        "split_mode": args.mode,
        "dataset": args.dataset,
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


def metadata_for_wizard_stratified(args, global_training_records, labels, client_records, client_label_counts, edges):
    return {
        "split_mode": args.mode,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "alpha": None,
        "num_clients": int(args.num_clients),
        "dataset_path": None,
        "output_root": args.output_root,
        "source_root": args.source_root,
        "global_test_policy": "not_present_in_source_root",
        "client_size_policy": "preserve_existing_wizard_client_sizes_from_source_root",
        "stratification_policy": "heuristic_task_family_plus_instruction_length_bucket",
        "instruction_length_bucket_edges": [int(edge) for edge in edges],
        "global_training_size": int(len(global_training_records)),
        "global_test_size": 0,
        "client_sizes": [int(len(records)) for records in client_records],
        "global_label_counts": label_counter(labels),
        "client_label_counts": {
            str(client_id): {
                label: int(count)
                for label, count in sorted(client_label_counts[client_id].items())
            }
            for client_id in range(len(client_records))
        },
    }


def create_legacy_dirichlet_split(args):
    if args.dataset != "dolly":
        raise ValueError("dirichlet mode is currently supported only for --dataset dolly.")

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
    if args.dataset == "dolly":
        create_dolly_stratified_split(args)
    elif args.dataset == "wizard":
        create_wizard_stratified_split(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_dolly_stratified_split(args):
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

    metadata = metadata_for_dolly_stratified(
        args=args,
        global_training_records=global_training_records,
        global_test_records=global_test_records,
        client_records=client_records,
        client_category_counts=client_category_counts,
    )
    write_json(output_dir / "split_metadata.json", metadata)


def read_source_clients(source_dir, num_clients):
    client_records = []
    for client_id in range(num_clients):
        client_path = source_dir / f"local_training_{client_id}.json"
        if not client_path.exists():
            raise FileNotFoundError(f"Could not find source client split {client_path}.")
        client_records.append(load_json(client_path))
    return client_records


def create_wizard_stratified_split(args):
    source_dir = Path(args.source_root) / str(args.num_clients)
    output_dir = make_output_dir(args.output_root, args.num_clients)

    if source_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "For stratified_keep_sizes, output-root must differ from source-root so the legacy split stays untouched."
        )

    source_client_records = read_source_clients(source_dir, args.num_clients)
    client_quotas = [len(records) for records in source_client_records]
    global_training_records = [
        record
        for client_records in source_client_records
        for record in client_records
    ]
    labels, edges = wizard_stratification_labels(
        global_training_records,
        args.num_length_buckets,
    )

    client_records, client_label_counts = build_stratified_clients(
        global_training_records=global_training_records,
        labels=labels,
        client_quotas=client_quotas,
        seed=args.seed,
    )

    for client_id, records in enumerate(client_records):
        write_json(output_dir / f"local_training_{client_id}.json", records)

    for filename in ("global_training.json", "global_test.json"):
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, output_dir / filename)

    metadata = metadata_for_wizard_stratified(
        args=args,
        global_training_records=global_training_records,
        labels=labels,
        client_records=client_records,
        client_label_counts=client_label_counts,
        edges=edges,
    )
    write_json(output_dir / "split_metadata.json", metadata)


def main():
    args = parse_args()
    random.seed(args.seed)
    if args.dataset == "dolly":
        require_numpy().random.seed(args.seed)

    if args.mode == "dirichlet":
        create_legacy_dirichlet_split(args)
    else:
        create_stratified_split(args)

    print(f"{args.dataset.capitalize()} split written to {Path(args.output_root) / str(args.num_clients)}")


if __name__ == "__main__":
    main()
