"""
Partition the CoLA training set across federated clients.

Usage:
    python client_data_allocation_cola.py <num_clients> <diff_quantity> [--alpha 0.5]

    diff_quantity=0  → IID (equal random shards)
    diff_quantity=1  → non-IID (Dirichlet distribution over labels)
"""

import sys
import numpy as np
import random
import os
import json
from datasets import load_dataset

num_clients = int(sys.argv[1])
diff_quantity = int(sys.argv[2])
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

np.random.seed(42)
random.seed(42)

# Load CoLA from GLUE benchmark
dataset = load_dataset("glue", "cola")
train_data = dataset["train"]
val_data = dataset["validation"]

print(f"CoLA train size: {len(train_data)}, validation size: {len(val_data)}")
print(f"Label distribution: {np.bincount([ex['label'] for ex in train_data])}")

data_path = os.path.join("data_cola", str(num_clients))
os.makedirs(data_path, exist_ok=True)

# Save validation set for global evaluation
val_records = [{"sentence": ex["sentence"], "label": ex["label"], "idx": ex["idx"]} for ex in val_data]
with open(os.path.join(data_path, "global_val.json"), "w") as f:
    json.dump(val_records, f)

# Convert train data to list of dicts
train_records = [{"sentence": ex["sentence"], "label": ex["label"], "idx": ex["idx"]} for i, ex in enumerate(train_data)]
train_labels = np.array([ex["label"] for ex in train_data])

if diff_quantity:
    # Non-IID: Dirichlet distribution over labels
    print(f"Non-IID partitioning with Dirichlet alpha={alpha}")
    min_size = 0
    min_require_size = 10
    N = len(train_records)
    unique_labels = np.unique(train_labels).tolist()

    while min_size < min_require_size:
        idx_partition = [[] for _ in range(num_clients)]
        for label in unique_labels:
            label_indices = np.where(train_labels == label)[0]
            np.random.shuffle(label_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balance: avoid giving too much to already-large partitions
            proportions = np.array(
                [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            idx_partition = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_partition, np.split(label_indices, split_points))
            ]
            min_size = min([len(idx_j) for idx_j in idx_partition])
        print(f"  min partition size: {min_size}")
else:
    # IID: equal random shards
    print("IID partitioning (equal random shards)")
    all_indices = list(range(len(train_records)))
    random.shuffle(all_indices)
    shard_size = len(all_indices) // num_clients
    idx_partition = []
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size if i < num_clients - 1 else len(all_indices)
        idx_partition.append(all_indices[start:end])

# Save client data
for client_id, indices in enumerate(idx_partition):
    client_records = [train_records[i] for i in indices]
    client_labels = [r["label"] for r in client_records]
    print(f"  Client_{client_id}: {len(client_records)} samples, "
          f"label dist: {np.bincount(client_labels, minlength=2).tolist()}")
    with open(os.path.join(data_path, f"local_training_{client_id}.json"), "w") as f:
        json.dump(client_records, f)

print(f"\nData saved to {data_path}/")
print(f"  - global_val.json ({len(val_records)} samples)")
print(f"  - local_training_{{0..{num_clients-1}}}.json")
