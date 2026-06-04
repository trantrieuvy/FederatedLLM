"""Local train/validation monitoring for federated clients."""

import json
import math
import os
from typing import Dict, Optional

import transformers


LOCAL_METRICS_FILENAME = "local_metrics.jsonl"


def initialize_local_metrics_file(output_dir: str, reset: bool = True) -> str:
    """Create or resume the durable metrics file for one monitored run."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, LOCAL_METRICS_FILENAME)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(metrics_path, "w" if reset else "a"):
            pass
    return metrics_path


def _finite_float(value) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def loss_to_perplexity(loss) -> Optional[float]:
    loss_value = _finite_float(loss)
    if loss_value is None:
        return None
    try:
        perplexity = math.exp(loss_value)
    except OverflowError:
        return None
    return perplexity if math.isfinite(perplexity) else None


def build_local_metric_record(
    *,
    method: str,
    round_id: int,
    client_id: int,
    local_epoch: int,
    train_on_inputs: Optional[bool],
    validation_source: str = "local_holdout",
    train_monitor_source: str = "capped_local_train",
    local_train_examples: int,
    monitored_train_examples: int,
    validation_examples: int,
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
) -> Dict[str, object]:
    train_loss = _finite_float(train_metrics.get("monitor_train_loss"))
    eval_loss = _finite_float(validation_metrics.get("eval_loss"))
    train_accuracy = _finite_float(train_metrics.get("monitor_train_accuracy"))
    eval_accuracy = _finite_float(validation_metrics.get("eval_accuracy"))
    return {
        "method": method,
        "round": int(round_id),
        "client_id": int(client_id),
        "local_epoch": int(local_epoch),
        "evaluation_point": "baseline" if int(local_epoch) == 0 else "local_epoch_end",
        "train_on_inputs": None if train_on_inputs is None else bool(train_on_inputs),
        "validation_source": validation_source,
        "train_monitor_source": train_monitor_source,
        "local_train_examples": int(local_train_examples),
        "monitored_train_examples": int(monitored_train_examples),
        "validation_examples": int(validation_examples),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "train_perplexity": loss_to_perplexity(train_loss),
        "eval_perplexity": loss_to_perplexity(eval_loss),
    }


def append_local_metric_record(metrics_path: str, record: Dict[str, object]) -> None:
    with open(metrics_path, "a") as handle:
        json.dump(record, handle, allow_nan=False)
        handle.write("\n")


def truncate_local_metrics_from_round(metrics_path: str, start_round: int) -> None:
    """Drop records from an uncommitted round before resuming a run."""
    if not os.path.exists(metrics_path):
        return
    retained_records = []
    with open(metrics_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["round"]) < int(start_round):
                retained_records.append(record)
    with open(metrics_path, "w") as handle:
        for record in retained_records:
            json.dump(record, handle, allow_nan=False)
            handle.write("\n")


class LocalMetricsTrainer(transformers.Trainer):
    """Trainer that records comparable local train/validation losses per epoch."""

    def __init__(
        self,
        *args,
        local_monitor_train_dataset,
        local_metrics_path: str,
        local_method: str,
        local_client_id: int,
        local_validation_source: str = "local_holdout",
        local_train_monitor_source: str = "capped_local_train",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.local_monitor_train_dataset = local_monitor_train_dataset
        self.local_metrics_path = local_metrics_path
        self.local_method = local_method
        self.local_client_id = int(local_client_id)
        self.local_validation_source = local_validation_source
        self.local_train_monitor_source = local_train_monitor_source
        self.local_monitor_context = None

    def evaluate_local_baseline(self, round_id: int, train_on_inputs: Optional[bool] = None) -> None:
        self.local_monitor_context = {
            "round_id": int(round_id),
            "train_on_inputs": None if train_on_inputs is None else bool(train_on_inputs),
        }
        self.evaluate()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        validation_metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        should_record = (
            self.local_monitor_context is not None
            and eval_dataset is None
            and metric_key_prefix == "eval"
        )
        if not should_record:
            return validation_metrics

        train_metrics = super().evaluate(
            eval_dataset=self.local_monitor_train_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix="monitor_train",
        )
        local_epoch = int(round(float(self.state.epoch or 0)))
        record = build_local_metric_record(
            method=self.local_method,
            round_id=self.local_monitor_context["round_id"],
            client_id=self.local_client_id,
            local_epoch=local_epoch,
            train_on_inputs=self.local_monitor_context["train_on_inputs"],
            validation_source=self.local_validation_source,
            train_monitor_source=self.local_train_monitor_source,
            local_train_examples=len(self.train_dataset),
            monitored_train_examples=len(self.local_monitor_train_dataset),
            validation_examples=len(self.eval_dataset),
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )
        if self.is_world_process_zero():
            append_local_metric_record(self.local_metrics_path, record)
            print(
                "  Local metrics "
                f"round={record['round']} client={record['client_id']} "
                f"local_epoch={record['local_epoch']} "
                f"train_loss={record['train_loss']} eval_loss={record['eval_loss']} "
                f"train_acc={record['train_accuracy']} eval_acc={record['eval_accuracy']}"
            )
        return validation_metrics
