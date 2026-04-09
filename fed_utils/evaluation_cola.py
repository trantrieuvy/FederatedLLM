"""
Global evaluation for CoLA: Matthews Correlation Coefficient on the validation set.
"""

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from datasets import Dataset


def global_evaluation_cola(model, tokenizer, val_data_path, max_seq_length=128, batch_size=64):
    """
    Evaluate model on CoLA validation set using Matthews Correlation Coefficient.

    Args:
        model: PEFT-wrapped RoBERTa model for sequence classification
        tokenizer: RoBERTa tokenizer
        val_data_path: path to global_val.json
        max_seq_length: max token length
        batch_size: eval batch size

    Returns:
        mcc: Matthews Correlation Coefficient (float)
    """
    import json
    with open(val_data_path, "r") as f:
        val_records = json.load(f)

    sentences = [r["sentence"] for r in val_records]
    labels = [r["label"] for r in val_records]

    # Tokenize
    encodings = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    dataset.set_format("torch")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    mcc = matthews_corrcoef(all_labels, all_preds)
    model.train()
    return mcc
