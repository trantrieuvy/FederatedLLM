"""
Federated adapter fine-tuning for RoBERTa on GLUE sequence-classification tasks.

This runner is intentionally narrow and RTE-ready: it supports the adapter
methods used by the epoch/round tuning manifests:

  * flora: normal linear FLoRA, with each round's stacked residual merged into
    the backbone weights
  * linear_flora_cumulative: linear stacked residuals retained across rounds
  * nonlinear_flora: stacked B GELU(Ax) residuals retained across rounds
  * ffa: frozen-A LoRA with trainable/averaged B only

The RoBERTa backbone is frozen. Each client trains fresh adapter parameters and
the classification head; the server aggregates adapters and the classifier with
dataset-size weights after every communication round.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from fed_utils.client_participation_scheduling import client_selection
from fed_utils.local_monitoring import (
    LocalMetricsTrainer,
    initialize_local_metrics_file,
    truncate_local_metrics_from_round,
)


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
SERVER_STATE_FILENAME = "server_state.pt"
LOCAL_VALIDATION_SOURCES = {"local_holdout", "global_val"}
RESIDUAL_METHODS = {"flora", "linear_flora_cumulative", "nonlinear_flora"}
CUMULATIVE_RESIDUAL_METHODS = {"linear_flora_cumulative", "nonlinear_flora"}
SUPPORTED_METHODS = RESIDUAL_METHODS | {"ffa"}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _set_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _load_json_records(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)


def _records_to_dataset(records: list[dict], sentence1_key: str, sentence2_key: Optional[str]) -> Dataset:
    payload = {
        sentence1_key: [record[sentence1_key] for record in records],
        "label": [int(record["label"]) for record in records],
    }
    if sentence2_key is not None:
        payload[sentence2_key] = [record[sentence2_key] for record in records]
    return Dataset.from_dict(payload)


def _target_linear_modules(model: nn.Module, target_modules: List[str]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            yield name, module


def _set_submodule(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _apply_activation(x: torch.Tensor, adapter_kind: str) -> torch.Tensor:
    if adapter_kind == "linear":
        return x
    if adapter_kind == "nonlinear":
        return F.gelu(x)
    raise ValueError(f"Unknown adapter kind: {adapter_kind}")


def _adapter_kind_for_method(method: str) -> str:
    if method in {"flora", "linear_flora_cumulative"}:
        return "linear"
    if method == "nonlinear_flora":
        return "nonlinear"
    raise ValueError(f"Method {method!r} does not use residual adapters.")


def _adapter_semantics_for_method(method: str) -> str:
    if method == "flora":
        return "merged_linear_residual"
    if method == "linear_flora_cumulative":
        return "cumulative_linear_residual"
    if method == "nonlinear_flora":
        return "cumulative_nonlinear_residual"
    if method == "ffa":
        return "ffa_global_B"
    raise ValueError(f"Unknown method: {method}")


def _resolve_resume_behavior_method(method: str, server_state: dict) -> Tuple[str, bool]:
    stored_semantics = server_state.get("adapter_semantics")
    legacy_flora_cumulative_state = (
        method == "flora"
        and (
            stored_semantics in {"stacked_residual", "cumulative_linear_residual"}
            or (stored_semantics is None and server_state.get("A_cumulative") is not None)
        )
    )
    if legacy_flora_cumulative_state:
        return "linear_flora_cumulative", True

    expected_semantics = _adapter_semantics_for_method(method)
    if stored_semantics is not None and stored_semantics != expected_semantics:
        raise ValueError(
            f"Resume state adapter semantics mismatch: "
            f"found {stored_semantics!r}, expected {expected_semantics!r}"
        )
    return method, False


class ResidualLoRALayer(nn.Module):
    """Linear or nonlinear LoRA block with optional frozen stacked residuals."""

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        adapter_kind: str,
        dropout: float = 0.0,
        init_std: float = 0.02,
        A_frozen: Optional[torch.Tensor] = None,
        B_frozen: Optional[torch.Tensor] = None,
        frozen_scaling: Optional[float] = None,
        train_new: bool = True,
    ):
        super().__init__()
        if (A_frozen is None) != (B_frozen is None):
            raise ValueError("A_frozen and B_frozen must be provided together.")
        if A_frozen is not None and frozen_scaling is None:
            raise ValueError("frozen_scaling is required when frozen adapters exist.")

        self.linear = linear
        self.adapter_kind = adapter_kind
        self.new_scaling = float(alpha) / int(r)
        self.frozen_scaling = frozen_scaling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A_new = nn.Parameter(torch.empty(r, linear.in_features, dtype=torch.float32))
        nn.init.normal_(self.A_new, mean=0.0, std=init_std)
        self.B_new = nn.Parameter(torch.zeros(linear.out_features, r, dtype=torch.float32))

        if A_frozen is not None:
            self.register_buffer("A_frozen", A_frozen.detach().clone().to(dtype=torch.float32))
            self.register_buffer("B_frozen", B_frozen.detach().clone().to(dtype=torch.float32))
        else:
            self.A_frozen = None
            self.B_frozen = None

        for parameter in self.linear.parameters():
            parameter.requires_grad = False
        if not train_new:
            self.A_new.requires_grad = False
            self.B_new.requires_grad = False

    def _adapter_update(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x.to(dtype=A.dtype), A)
        hidden = _apply_activation(hidden, self.adapter_kind)
        return F.linear(hidden.to(dtype=B.dtype), B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.A_frozen is not None:
            frozen_update = self._adapter_update(x, self.A_frozen, self.B_frozen)
            y = y + (self.frozen_scaling * frozen_update).to(dtype=y.dtype)

        new_update = self._adapter_update(self.dropout(x), self.A_new, self.B_new)
        return y + (self.new_scaling * new_update).to(dtype=y.dtype)


class FFALoRALayer(nn.Module):
    """Frozen-A LoRA block. Only B is trainable."""

    def __init__(
        self,
        linear: nn.Linear,
        A_frozen: torch.Tensor,
        B_initial: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.linear = linear
        self.scaling = float(scaling)
        self.activation = (activation or "none").lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("A_frozen", A_frozen.detach().clone().to(dtype=torch.float32))
        self.B = nn.Parameter(B_initial.detach().clone().to(dtype=torch.float32))
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ("none", "linear", "identity"):
            return x
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unknown FFA activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        hidden = F.linear(self.dropout(x).to(dtype=self.A_frozen.dtype), self.A_frozen)
        hidden = self._activate(hidden)
        update = F.linear(hidden.to(dtype=self.B.dtype), self.B)
        return y + (self.scaling * update).to(dtype=y.dtype)


def _inject_residual_adapters(
    base_model: nn.Module,
    target_modules: List[str],
    r: int,
    alpha: float,
    adapter_kind: str,
    dropout: float,
    A_frozen_dict: Optional[Dict[str, torch.Tensor]],
    B_frozen_dict: Optional[Dict[str, torch.Tensor]],
    frozen_scaling: float,
    train_new: bool = True,
) -> Tuple[nn.Module, int]:
    count = 0
    for name, module in list(_target_linear_modules(base_model, target_modules)):
        adapter = ResidualLoRALayer(
            module,
            r=r,
            alpha=alpha,
            adapter_kind=adapter_kind,
            dropout=dropout,
            A_frozen=A_frozen_dict.get(name) if A_frozen_dict else None,
            B_frozen=B_frozen_dict.get(name) if B_frozen_dict else None,
            frozen_scaling=frozen_scaling,
            train_new=train_new,
        ).to(module.weight.device)
        _set_submodule(base_model, name, adapter)
        count += 1
    return base_model, count


def _init_frozen_A(
    model: nn.Module,
    target_modules: List[str],
    r: int,
    seed: int,
    init_std: float,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return {
        name: torch.randn(r, module.in_features, generator=generator, dtype=torch.float32) * init_std
        for name, module in _target_linear_modules(model, target_modules)
    }


def _init_zero_B(model: nn.Module, target_modules: List[str], r: int) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros(module.out_features, r, dtype=torch.float32)
        for name, module in _target_linear_modules(model, target_modules)
    }


def _inject_ffa_adapters(
    base_model: nn.Module,
    target_modules: List[str],
    A_frozen_dict: Dict[str, torch.Tensor],
    B_dict: Dict[str, torch.Tensor],
    scaling: float,
    dropout: float,
    activation: str,
    client_r: Optional[int] = None,
) -> Tuple[nn.Module, int]:
    count = 0
    for name, module in list(_target_linear_modules(base_model, target_modules)):
        A = A_frozen_dict[name]
        B = B_dict[name]
        if client_r is not None:
            A = A[:client_r, :]
            B = B[:, :client_r]
        adapter = FFALoRALayer(
            module,
            A_frozen=A,
            B_initial=B,
            scaling=scaling,
            dropout=dropout,
            activation=activation,
        ).to(module.weight.device)
        _set_submodule(base_model, name, adapter)
        count += 1
    return base_model, count


def _set_trainable_parameters(model: nn.Module, method: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        if name.startswith("classifier."):
            parameter.requires_grad = True
        elif method in RESIDUAL_METHODS and (
            name.endswith(".A_new") or name.endswith(".B_new")
        ):
            parameter.requires_grad = True
        elif method == "ffa" and name.endswith(".B"):
            parameter.requires_grad = True


def _adapter_state_dict(model: nn.Module, method: str) -> Dict[str, torch.Tensor]:
    suffixes = (".A_new", ".B_new") if method in RESIDUAL_METHODS else (".B",)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith(suffixes)
    }


def _classifier_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }


def _load_classifier_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    for name, value in state_dict.items():
        if name not in named_parameters:
            raise KeyError(f"Classifier parameter not found: {name}")
        named_parameters[name].data.copy_(value.to(device=named_parameters[name].device, dtype=named_parameters[name].dtype))


def _model_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _module_state_from_adapter_state(
    adapter_state: Dict[str, torch.Tensor],
    *,
    A_suffix: str,
    B_suffix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    A_dict: Dict[str, torch.Tensor] = {}
    B_dict: Dict[str, torch.Tensor] = {}
    for key, value in adapter_state.items():
        if key.endswith(A_suffix):
            A_dict[key[: -len(A_suffix)]] = value.detach().cpu().clone()
        elif key.endswith(B_suffix):
            B_dict[key[: -len(B_suffix)]] = value.detach().cpu().clone()
    return A_dict, B_dict


def _validate_A_B(A_dict: Dict[str, torch.Tensor], B_dict: Dict[str, torch.Tensor], label: str) -> None:
    if set(A_dict) != set(B_dict):
        raise ValueError(
            f"{label} A/B keys differ: missing_A={sorted(set(B_dict) - set(A_dict))}, "
            f"missing_B={sorted(set(A_dict) - set(B_dict))}"
        )
    for name in A_dict:
        A = A_dict[name]
        B = B_dict[name]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError(f"{label} adapter {name} has incompatible shapes A{tuple(A.shape)} B{tuple(B.shape)}")


def _aggregate_classifier(
    client_states: list[Tuple[int, Dict[str, torch.Tensor]]],
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    aggregated: Optional[Dict[str, torch.Tensor]] = None
    for idx, (_, state_dict) in enumerate(client_states):
        weight = weights[idx]
        if aggregated is None:
            aggregated = {
                key: value.to(dtype=torch.float32) * weight
                for key, value in state_dict.items()
            }
        else:
            for key, value in state_dict.items():
                aggregated[key] += value.to(dtype=torch.float32) * weight
    if aggregated is None:
        raise ValueError("Cannot aggregate classifier without client states.")
    return aggregated


def _aggregate_stacked_residuals(
    client_adapter_states: list[Tuple[int, Dict[str, torch.Tensor]]],
    weights: torch.Tensor,
    local_ranks: List[int],
    lora_r: int,
    heter: bool,
    nonlinear: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    A_stacked: Optional[Dict[str, torch.Tensor]] = None
    B_stacked: Optional[Dict[str, torch.Tensor]] = None

    for idx, (client_id, adapter_state) in enumerate(client_adapter_states):
        client_r = local_ranks[client_id] if heter else lora_r
        A_client, B_client = _module_state_from_adapter_state(
            adapter_state,
            A_suffix=".A_new",
            B_suffix=".B_new",
        )
        _validate_A_B(A_client, B_client, f"client {client_id}")
        weight = weights[idx]

        if A_stacked is None:
            A_stacked = {}
            B_stacked = {}
            for name in sorted(A_client):
                A_stacked[name] = A_client[name].to(dtype=torch.float32).clone()
                B_stacked[name] = B_client[name].to(dtype=torch.float32).clone()
                if nonlinear:
                    B_stacked[name] *= weight
                else:
                    A_stacked[name] *= weight
        else:
            assert B_stacked is not None
            for name in sorted(A_client):
                A = A_client[name].to(dtype=torch.float32)
                B = B_client[name].to(dtype=torch.float32)
                if A.shape[0] != client_r or B.shape[1] != client_r:
                    raise ValueError(
                        f"Client {client_id} rank mismatch for {name}: "
                        f"expected {client_r}, got A{tuple(A.shape)} B{tuple(B.shape)}"
                    )
                A_stacked[name] = torch.cat(
                    [A_stacked[name], A if nonlinear else A * weight],
                    dim=0,
                )
                B_stacked[name] = torch.cat(
                    [B_stacked[name], B * weight if nonlinear else B],
                    dim=1,
                )

    if A_stacked is None or B_stacked is None:
        raise ValueError("Cannot aggregate adapters without client states.")
    return A_stacked, B_stacked


def _append_stacked_residuals(
    A_prev: Optional[Dict[str, torch.Tensor]],
    B_prev: Optional[Dict[str, torch.Tensor]],
    A_round: Dict[str, torch.Tensor],
    B_round: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    _validate_A_B(A_round, B_round, "round")
    if A_prev is None:
        return (
            {name: tensor.detach().cpu().clone() for name, tensor in A_round.items()},
            {name: tensor.detach().cpu().clone() for name, tensor in B_round.items()},
        )
    if B_prev is None:
        raise ValueError("B_prev is missing while A_prev is present.")
    _validate_A_B(A_prev, B_prev, "previous")
    if set(A_prev) != set(A_round):
        raise ValueError("Cannot append adapter dictionaries with different module keys.")

    return (
        {name: torch.cat([A_prev[name], A_round[name]], dim=0).clone() for name in sorted(A_round)},
        {name: torch.cat([B_prev[name], B_round[name]], dim=1).clone() for name in sorted(B_round)},
    )


def _merge_linear_residual_into_model(
    model: nn.Module,
    A_round: Dict[str, torch.Tensor],
    B_round: Dict[str, torch.Tensor],
    scaling: float,
) -> None:
    _validate_A_B(A_round, B_round, "round")
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name in sorted(A_round):
            module = modules.get(name)
            if module is None:
                raise KeyError(f"Cannot merge residual: target module not found: {name}")
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Cannot merge residual into non-linear module {name}: {type(module).__name__}")
            A = A_round[name].to(device=module.weight.device, dtype=torch.float32)
            B = B_round[name].to(device=module.weight.device, dtype=torch.float32)
            delta = torch.matmul(B, A) * float(scaling)
            if delta.shape != module.weight.shape:
                raise ValueError(
                    f"Residual delta shape mismatch for {name}: "
                    f"delta{tuple(delta.shape)} vs weight{tuple(module.weight.shape)}"
                )
            module.weight.data.add_(delta.to(dtype=module.weight.dtype))


def _aggregate_ffa_B(
    client_adapter_states: list[Tuple[int, Dict[str, torch.Tensor]]],
    weights: torch.Tensor,
    global_B_template: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    aggregated = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in global_B_template.items()
    }

    for idx, (client_id, adapter_state) in enumerate(client_adapter_states):
        B_client = {
            key[:-2]: value.detach().cpu().to(dtype=torch.float32)
            for key, value in adapter_state.items()
            if key.endswith(".B")
        }
        weight = weights[idx]
        for name, client_B in B_client.items():
            if name not in aggregated:
                raise KeyError(f"Unexpected FFA B key from client {client_id}: {name}")
            target = aggregated[name]
            if client_B.shape == target.shape:
                target += client_B * weight
            elif client_B.ndim == 2 and target.ndim == 2 and client_B.shape[0] == target.shape[0]:
                target[:, : client_B.shape[1]] += client_B * weight
            else:
                raise ValueError(
                    f"Cannot aggregate FFA B for {name}: client shape {tuple(client_B.shape)}, "
                    f"target shape {tuple(target.shape)}"
                )
    return aggregated


def _write_residual_adapter_state(path: str | Path, A_dict: Dict[str, torch.Tensor], B_dict: Dict[str, torch.Tensor]) -> None:
    state = {}
    for name in sorted(A_dict):
        state[f"{name}.A"] = A_dict[name].detach().cpu().clone()
        state[f"{name}.B"] = B_dict[name].detach().cpu().clone()
    torch.save(state, path)


def _write_ffa_adapter_state(path: str | Path, B_dict: Dict[str, torch.Tensor]) -> None:
    torch.save({f"{name}.B": value.detach().cpu().clone() for name, value in B_dict.items()}, path)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _write_latest_server_state(path: Path, state: dict) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


class GlueClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_path: str | Path,
        output_dir: str | Path,
        sentence1_key: str,
        sentence2_key: Optional[str],
        seed: int,
        local_metrics_path: Optional[str] = None,
        global_eval_data: Optional[Dataset] = None,
    ):
        self.client_id = int(client_id)
        self.model = model
        self.output_dir = Path(output_dir)
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.seed = seed
        self.local_metrics_path = local_metrics_path
        self.local_monitoring_enabled = local_metrics_path is not None
        self.global_eval_data = global_eval_data
        local_data_path = Path(data_path) / f"local_training_{self.client_id}.json"
        self.local_data = _records_to_dataset(_load_json_records(local_data_path), sentence1_key, sentence2_key)

    def prepare_local_dataset(
        self,
        tokenizer,
        max_seq_length: int,
        local_val_set_size: float = 0,
        local_train_monitor_size: int = 500,
        local_validation_source: str = "local_holdout",
    ) -> None:
        local_validation_source = str(local_validation_source).strip().lower()
        if local_validation_source not in LOCAL_VALIDATION_SOURCES:
            raise ValueError(f"Unknown local_validation_source: {local_validation_source!r}.")
        if local_train_monitor_size < 0:
            raise ValueError("local_train_monitor_size must not be negative.")

        def tokenize_fn(examples):
            if self.sentence2_key is None:
                tokenized = tokenizer(
                    examples[self.sentence1_key],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                )
            else:
                tokenized = tokenizer(
                    examples[self.sentence1_key],
                    examples[self.sentence2_key],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                )
            tokenized["labels"] = examples["label"]
            return tokenized

        remove_columns = [self.sentence1_key, "label"]
        if self.sentence2_key is not None:
            remove_columns.append(self.sentence2_key)

        def tokenize_dataset(dataset: Dataset) -> Dataset:
            tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
            tokenized.set_format("torch")
            return tokenized

        split_seed = self.seed + self.client_id
        self.local_eval_dataset = None
        self.local_monitor_train_dataset = None
        self.local_validation_source = local_validation_source
        self.local_train_monitor_source = (
            "full_local_train" if int(local_train_monitor_size) == 0 else "capped_local_train"
        )
        if local_val_set_size > 0:
            local_split = self.local_data.train_test_split(
                test_size=local_val_set_size,
                seed=split_seed,
            )
            local_train = local_split["train"].shuffle(seed=split_seed)
            local_eval = local_split["test"].shuffle(seed=split_seed)
        else:
            local_train = self.local_data.shuffle(seed=split_seed)
            local_eval = None

        if self.local_monitoring_enabled:
            if local_validation_source == "local_holdout":
                if local_eval is None:
                    raise ValueError("Local holdout monitoring requires local_val_set_size > 0.")
                self.local_eval_dataset = tokenize_dataset(local_eval)
            else:
                if self.global_eval_data is None:
                    raise ValueError("Global validation monitoring requires global_eval_data.")
                self.local_eval_dataset = tokenize_dataset(self.global_eval_data)

            if int(local_train_monitor_size) == 0:
                monitor_train = local_train
            else:
                monitor_size = min(int(local_train_monitor_size), len(local_train))
                monitor_train = local_train.select(range(monitor_size))
            self.local_monitor_train_dataset = tokenize_dataset(monitor_train)

        self.local_train_dataset = tokenize_dataset(local_train)

    def build_trainer(
        self,
        tokenizer,
        method: str,
        local_micro_batch_size: int,
        gradient_accumulation_steps: int,
        local_num_epochs: int,
        local_learning_rate: float,
        warmup_ratio: float,
        weight_decay: float,
        fp16: bool,
        bf16: bool,
        ddp: bool,
        local_monitor_accuracy: bool = False,
    ) -> None:
        monitoring = self.local_monitoring_enabled and self.local_eval_dataset is not None
        train_args = TrainingArguments(
            output_dir=str(self.output_dir / "trainer_saved" / f"local_output_{self.client_id}"),
            overwrite_output_dir=True,
            per_device_train_batch_size=local_micro_batch_size,
            per_device_eval_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            logging_steps=50,
            evaluation_strategy="epoch" if monitoring else "no",
            save_strategy="no",
            optim="adamw_torch",
            report_to="none",
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False if ddp else None,
            seed=self.seed,
        )
        trainer_kwargs = dict(
            model=self.model,
            args=train_args,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            compute_metrics=_classification_accuracy_metrics if monitoring and local_monitor_accuracy else None,
        )
        if monitoring:
            self.trainer = LocalMetricsTrainer(
                **trainer_kwargs,
                local_monitor_train_dataset=self.local_monitor_train_dataset,
                local_metrics_path=self.local_metrics_path,
                local_method=method,
                local_client_id=self.client_id,
                local_validation_source=self.local_validation_source,
                local_train_monitor_source=self.local_train_monitor_source,
            )
        else:
            self.trainer = Trainer(**trainer_kwargs)

    def evaluate_local_baseline(self, round_id: int) -> None:
        if self.local_monitoring_enabled:
            self.trainer.evaluate_local_baseline(round_id)

    def train(self) -> None:
        self.trainer.train()

    def save_trainable_state(
        self,
        method: str,
        epoch: int,
        retain_output: bool = True,
    ) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        adapter_state = _adapter_state_dict(self.model, method)
        classifier_state = _classifier_state_dict(self.model)
        if retain_output:
            single_output_dir = self.output_dir / str(epoch) / f"local_output_{self.client_id}"
            single_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "adapter": adapter_state,
                    "classifier": classifier_state,
                    "num_examples": len(self.local_train_dataset),
                },
                single_output_dir / "pytorch_model.bin",
            )
        return len(self.local_train_dataset), adapter_state, classifier_state


def _evaluate_accuracy(
    model: nn.Module,
    tokenizer,
    records: list[dict],
    sentence1_key: str,
    sentence2_key: Optional[str],
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    labels = [int(record["label"]) for record in records]
    predictions: list[int] = []

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            if sentence2_key is None:
                encoded = tokenizer(
                    [record[sentence1_key] for record in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    [record[sentence1_key] for record in batch],
                    [record[sentence2_key] for record in batch],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    model.train()
    return correct / len(labels)


def _classification_accuracy_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


def _log_path_for_method(output_dir: str | Path, method: str, num_clients: int) -> Path:
    client_dir = Path(output_dir) / str(num_clients)
    if method == "flora":
        return Path(f"{client_dir}log.txt")
    return client_dir / "log.txt"


def _mirror_log_for_analysis(output_dir: str | Path, method: str, num_clients: int) -> None:
    """Write the alternate flora log path so old and new analysis both work."""
    if method != "flora":
        return
    legacy_path = _log_path_for_method(output_dir, method, num_clients)
    standard_path = Path(output_dir) / str(num_clients) / "log.txt"
    if legacy_path.exists():
        standard_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_path, standard_path)


def fl_finetune(
    # method / model / data
    method: str = "flora",
    global_model: str = "roberta-base",
    task_name: str = "rte",
    data_path: str = "./data_rte_stratified",
    output_dir: str = "./fed-roberta-rte/",
    max_seq_length: int = 512,
    # FL
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1.0,
    num_communication_rounds: int = 10,
    num_clients: int = 10,
    # local training
    local_batch_size: int = 32,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 1,
    local_learning_rate: float = 5e-4,
    local_val_set_size: float = 0,
    local_train_monitor_size: int = 500,
    local_validation_source: str = "local_holdout",
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.1,
    fp16: bool = False,
    bf16: bool = False,
    # adapters
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    lora_target_modules: List[str] = ["query", "value"],
    heter: bool = False,
    local_ranks: List[int] = [32, 16, 8, 8, 4, 4, 2, 2, 2, 2],
    activation: str = "gelu",
    A_init_std: float = 0.02,
    # misc
    eval_batch_size: int = 64,
    seed: int = 0,
    use_deterministic_algorithms: bool = True,
    resume_from_latest: bool = False,
    max_rounds_per_invocation: int = 0,
    retain_adapter_every_n_rounds: int = 1,
    local_monitor_accuracy: bool = False,
):
    method = method.lower()
    task_name = task_name.lower()
    heter = _as_bool(heter)
    resume_from_latest = _as_bool(resume_from_latest)
    local_monitor_accuracy = _as_bool(local_monitor_accuracy)
    use_deterministic_algorithms = _as_bool(use_deterministic_algorithms)
    local_validation_source = str(local_validation_source).strip().lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if task_name not in TASK_TO_KEYS:
        raise ValueError(f"Unknown GLUE task: {task_name}")
    if local_validation_source not in LOCAL_VALIDATION_SOURCES:
        raise ValueError(f"Unknown local_validation_source: {local_validation_source!r}.")
    if heter and len(local_ranks) < num_clients:
        raise ValueError("local_ranks must provide at least one rank per client.")
    if local_train_monitor_size < 0:
        raise ValueError("local_train_monitor_size must not be negative.")
    if max_rounds_per_invocation < 0:
        raise ValueError("max_rounds_per_invocation must not be negative.")
    if retain_adapter_every_n_rounds < 0:
        raise ValueError("retain_adapter_every_n_rounds must not be negative.")

    behavior_method = method
    legacy_cumulative_flora_resume = False

    data_root = Path(data_path) / str(num_clients)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing federated split: {data_root}")
    val_records = _load_json_records(data_root / "global_val.json")
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]

    base_scaling = float(lora_alpha) / int(lora_r)
    global_ffa_r = max(local_ranks[:num_clients]) if heter else lora_r
    gradient_accumulation_steps = max(local_batch_size // local_micro_batch_size, 1)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    print(
        "Federated RoBERTa GLUE tuning\n"
        f"  method:                  {method}\n"
        f"  adapter semantics:       {_adapter_semantics_for_method(method)}\n"
        f"  task_name:               {task_name}\n"
        f"  global_model:            {global_model}\n"
        f"  data_path:               {data_root}\n"
        f"  rounds:                  {num_communication_rounds}\n"
        f"  local_num_epochs:        {local_num_epochs}\n"
        f"  local_batch_size:        {local_batch_size}\n"
        f"  local_learning_rate:     {local_learning_rate}\n"
        f"  local_val_set_size:      {local_val_set_size}\n"
        f"  local_train_monitor_size: {local_train_monitor_size}\n"
        f"  local_validation_source: {local_validation_source}\n"
        f"  lora_r/lora_alpha:       {lora_r}/{lora_alpha}\n"
        f"  effective scaling:       {base_scaling}\n"
        f"  heter:                   {heter}\n"
        f"  local_ranks:             {local_ranks[:num_clients] if heter else None}\n"
        f"  seed:                    {seed}\n"
        f"  resume_from_latest:      {resume_from_latest}\n"
        f"  max_rounds_per_invocation: {max_rounds_per_invocation}\n"
        f"  retain_adapter_every_n_rounds: {retain_adapter_every_n_rounds}\n"
        f"  local_monitor_accuracy:  {local_monitor_accuracy}\n"
    )

    _set_seed(seed, use_deterministic_algorithms)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(global_model, num_labels=2, finetuning_task=task_name)
    raw_model = AutoModelForSequenceClassification.from_pretrained(global_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=True)

    for parameter in raw_model.parameters():
        parameter.requires_grad = False
    raw_model.to("cpu")

    A_cumulative: Optional[Dict[str, torch.Tensor]] = None
    B_cumulative: Optional[Dict[str, torch.Tensor]] = None
    A_ffa: Optional[Dict[str, torch.Tensor]] = None
    B_ffa: Optional[Dict[str, torch.Tensor]] = None
    if method == "ffa":
        A_ffa = _init_frozen_A(raw_model, lora_target_modules, global_ffa_r, seed, A_init_std)
        B_ffa = _init_zero_B(raw_model, lora_target_modules, global_ffa_r)

    output_client_dir = Path(output_dir) / str(num_clients)
    output_client_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_client_dir / SERVER_STATE_FILENAME
    accuracy_list: list[float] = []
    expected_cumulative_residual_r = 0
    start_round = 0
    if resume_from_latest:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume without server state: {state_path}")
        server_state = torch.load(state_path, map_location="cpu")
        expected_state = {
            "method": method,
            "num_clients": int(num_clients),
            "num_communication_rounds": int(num_communication_rounds),
            "lora_r": int(lora_r),
            "lora_alpha": float(lora_alpha),
            "heter": bool(heter),
            "seed": int(seed),
        }
        for name, expected_value in expected_state.items():
            if server_state.get(name) != expected_value:
                raise ValueError(
                    f"Resume state mismatch for {name}: "
                    f"found {server_state.get(name)!r}, expected {expected_value!r}"
                )
        behavior_method, legacy_cumulative_flora_resume = _resolve_resume_behavior_method(
            method,
            server_state,
        )
        if legacy_cumulative_flora_resume:
            print(
                "Resuming legacy RoBERTa flora state as Linear FLoRA Cumulative. "
                "Fresh --method flora runs use normal merged FLoRA semantics."
            )
        if behavior_method == "flora":
            raw_model_state = server_state.get("raw_model_state")
            if raw_model_state is None:
                raise ValueError("Cannot resume normal flora without raw_model_state in server_state.pt.")
            raw_model.load_state_dict(raw_model_state)
        else:
            _load_classifier_state(raw_model, server_state["classifier_state"])
        if behavior_method in CUMULATIVE_RESIDUAL_METHODS:
            A_cumulative = server_state.get("A_cumulative")
            B_cumulative = server_state.get("B_cumulative")
        B_ffa = server_state.get("B_ffa", B_ffa)
        accuracy_list = [float(value) for value in server_state["accuracy_list"]]
        expected_cumulative_residual_r = int(server_state["expected_cumulative_residual_r"])
        start_round = int(server_state["completed_round"]) + 1
        _restore_rng_state(server_state["rng_state"])

    if start_round >= num_communication_rounds:
        print(f"All {num_communication_rounds} communication rounds are already complete.")
        return

    end_round = num_communication_rounds
    if max_rounds_per_invocation > 0:
        end_round = min(start_round + max_rounds_per_invocation, num_communication_rounds)

    local_metrics_enabled = local_val_set_size > 0 or local_validation_source == "global_val"
    local_metrics_path = (
        initialize_local_metrics_file(str(output_client_dir), reset=not resume_from_latest)
        if local_metrics_enabled
        else None
    )
    if local_metrics_path is not None and resume_from_latest:
        truncate_local_metrics_from_round(local_metrics_path, start_round)
    log_path = _log_path_for_method(output_dir, method, num_clients)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Starting federated RoBERTa training. "
        f"Active adapter semantics: {_adapter_semantics_for_method(behavior_method)}. "
        f"Executing rounds {start_round} through {end_round - 1} "
        f"of {num_communication_rounds - 1}."
    )
    for epoch in tqdm(range(start_round, end_round)):
        print(f"\n=== Round {epoch} ===")
        retain_round_artifacts = (
            epoch == num_communication_rounds - 1
            or (
                retain_adapter_every_n_rounds > 0
                and (epoch + 1) % retain_adapter_every_n_rounds == 0
            )
        )
        selected_clients = sorted(
            client_selection(
                num_clients,
                client_selection_frac,
                client_selection_strategy,
                other_info=epoch,
            )
        )
        print(f"  Selected clients: {selected_clients}")

        client_adapter_states: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        client_classifier_states: list[Tuple[int, Dict[str, torch.Tensor]]] = []
        dataset_sizes: list[int] = []

        for client_id in selected_clients:
            client_r = local_ranks[client_id] if heter else lora_r
            client_alpha = base_scaling * client_r
            raw_model.to("cpu")
            model_client = AutoModelForSequenceClassification.from_config(config)
            model_client.load_state_dict(raw_model.state_dict())

            if behavior_method in RESIDUAL_METHODS:
                adapter_kind = _adapter_kind_for_method(behavior_method)
                use_cumulative_residual = behavior_method in CUMULATIVE_RESIDUAL_METHODS
                model_client, n_adapters = _inject_residual_adapters(
                    model_client,
                    target_modules=lora_target_modules,
                    r=client_r,
                    alpha=client_alpha,
                    adapter_kind=adapter_kind,
                    dropout=lora_dropout,
                    A_frozen_dict=A_cumulative if use_cumulative_residual else None,
                    B_frozen_dict=B_cumulative if use_cumulative_residual else None,
                    frozen_scaling=base_scaling,
                    train_new=True,
                )
            else:
                assert A_ffa is not None and B_ffa is not None
                model_client, n_adapters = _inject_ffa_adapters(
                    model_client,
                    target_modules=lora_target_modules,
                    A_frozen_dict=A_ffa,
                    B_dict=B_ffa,
                    scaling=base_scaling,
                    dropout=lora_dropout,
                    activation=activation,
                    client_r=client_r if heter else None,
                )

            _set_trainable_parameters(model_client, behavior_method)
            model_client.to(device)
            trainable = sum(parameter.numel() for parameter in model_client.parameters() if parameter.requires_grad)
            print(f"  Client_{client_id}: rank={client_r}, adapters={n_adapters}, trainable={trainable:,}")

            client = GlueClient(
                client_id,
                model_client,
                data_root,
                output_client_dir,
                sentence1_key,
                sentence2_key,
                seed,
                local_metrics_path=local_metrics_path,
                global_eval_data=_records_to_dataset(val_records, sentence1_key, sentence2_key),
            )
            client.prepare_local_dataset(
                tokenizer,
                max_seq_length,
                local_val_set_size,
                local_train_monitor_size,
                local_validation_source,
            )
            client.build_trainer(
                tokenizer,
                method,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                warmup_ratio,
                weight_decay,
                fp16,
                bf16,
                ddp,
                local_monitor_accuracy=local_monitor_accuracy,
            )
            client.evaluate_local_baseline(epoch)
            client.train()
            num_examples, adapter_state, classifier_state = client.save_trainable_state(
                behavior_method,
                epoch,
                retain_output=retain_round_artifacts,
            )
            dataset_sizes.append(num_examples)
            client_adapter_states.append((client_id, adapter_state))
            client_classifier_states.append((client_id, classifier_state))
            del client, model_client
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weights = torch.tensor(dataset_sizes, dtype=torch.float32)
        weights = weights / weights.sum()
        print(f"  Aggregation weights: {[round(float(weight), 6) for weight in weights]}")

        classifier_state = _aggregate_classifier(client_classifier_states, weights)
        _load_classifier_state(raw_model, classifier_state)

        round_dir = output_client_dir / str(epoch)
        round_dir.mkdir(parents=True, exist_ok=True)
        expected_round_r = (
            sum(local_ranks[client_id] for client_id in selected_clients)
            if heter
            else len(selected_clients) * lora_r
        )

        if behavior_method in RESIDUAL_METHODS:
            nonlinear = behavior_method == "nonlinear_flora"
            A_round, B_round = _aggregate_stacked_residuals(
                client_adapter_states,
                weights,
                local_ranks,
                lora_r,
                heter,
                nonlinear,
            )
            round_r = next(iter(A_round.values())).shape[0]
            if behavior_method == "flora":
                _merge_linear_residual_into_model(raw_model, A_round, B_round, base_scaling)
                if retain_round_artifacts:
                    _write_residual_adapter_state(round_dir / "adapter_model.bin", A_round, B_round)
                stacked_r = round_r
                expected_global_r = expected_round_r
            else:
                A_cumulative, B_cumulative = _append_stacked_residuals(
                    A_cumulative,
                    B_cumulative,
                    A_round,
                    B_round,
                )
                if retain_round_artifacts:
                    _write_residual_adapter_state(round_dir / "adapter_model_delta.bin", A_round, B_round)
                    _write_residual_adapter_state(round_dir / "adapter_model.bin", A_cumulative, B_cumulative)
                stacked_r = next(iter(A_cumulative.values())).shape[0]
                expected_cumulative_residual_r += expected_round_r
                expected_global_r = expected_cumulative_residual_r
        else:
            assert B_ffa is not None
            B_ffa = _aggregate_ffa_B(client_adapter_states, weights, B_ffa)
            if retain_round_artifacts:
                _write_ffa_adapter_state(round_dir / "adapter_model.bin", B_ffa)
            stacked_r = next(iter(B_ffa.values())).shape[1]
            round_r = stacked_r
            expected_global_r = global_ffa_r

        expected_reported_round_r = expected_round_r if behavior_method in RESIDUAL_METHODS else global_ffa_r
        if round_r != expected_reported_round_r or stacked_r != expected_global_r:
            raise ValueError(
                f"Rank sanity check failed at round {epoch}: "
                f"round_r={round_r} expected={expected_reported_round_r}, "
                f"cumulative_or_global_r={stacked_r} expected={expected_global_r}"
            )
        print(
            "  Rank check: "
            f"round_r={round_r} expected={expected_reported_round_r}; "
            f"cumulative_or_global_r={stacked_r} expected={expected_global_r}"
        )

        eval_model = AutoModelForSequenceClassification.from_config(config)
        eval_model.load_state_dict(raw_model.state_dict())
        if behavior_method in CUMULATIVE_RESIDUAL_METHODS:
            assert A_cumulative is not None and B_cumulative is not None
            eval_model, _ = _inject_residual_adapters(
                eval_model,
                target_modules=lora_target_modules,
                r=lora_r,
                alpha=lora_alpha,
                adapter_kind=_adapter_kind_for_method(behavior_method),
                dropout=0.0,
                A_frozen_dict=A_cumulative,
                B_frozen_dict=B_cumulative,
                frozen_scaling=base_scaling,
                train_new=False,
            )
        elif method == "ffa":
            assert A_ffa is not None and B_ffa is not None
            eval_model, _ = _inject_ffa_adapters(
                eval_model,
                target_modules=lora_target_modules,
                A_frozen_dict=A_ffa,
                B_dict=B_ffa,
                scaling=base_scaling,
                dropout=0.0,
                activation=activation,
                client_r=None,
            )

        eval_model.to(device)
        accuracy = _evaluate_accuracy(
            eval_model,
            tokenizer,
            val_records,
            sentence1_key,
            sentence2_key,
            max_seq_length,
            eval_batch_size,
            device,
        )
        accuracy_list.append(accuracy)
        print(f"  Acc round {epoch}: {accuracy}")
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(round_dir / "round_config.json", "w") as handle:
            json.dump(
                {
                    "epoch": int(epoch),
                    "method": method,
                    "effective_method": behavior_method,
                    "task_name": task_name,
                    "global_model": global_model,
                    "lora_r": int(lora_r),
                    "lora_alpha": float(lora_alpha),
                    "effective_scaling": float(base_scaling),
                    "heter": bool(heter),
                    "local_ranks": [int(rank) for rank in local_ranks[:num_clients]] if heter else None,
                    "selected_clients": [int(client_id) for client_id in selected_clients],
                    "local_dataset_sizes": {
                        int(client_id): int(size)
                        for client_id, size in zip(selected_clients, dataset_sizes)
                    },
                    "local_val_set_size": local_val_set_size,
                    "local_train_monitor_size": int(local_train_monitor_size),
                    "local_validation_source": local_validation_source,
                    "local_train_monitor_source": (
                        "full_local_train" if int(local_train_monitor_size) == 0 else "capped_local_train"
                    ),
                    "local_metrics_path": str(local_metrics_path) if local_metrics_path else None,
                    "adapter_artifacts_retained": bool(retain_round_artifacts),
                    "server_state_path": str(state_path),
                    "round_stacked_r": int(round_r),
                    "expected_round_stacked_r": int(expected_reported_round_r),
                    "cumulative_or_global_r": int(stacked_r),
                    "expected_cumulative_or_global_r": int(expected_global_r),
                    "rank_semantics": _adapter_semantics_for_method(behavior_method),
                    "legacy_cumulative_flora_resume": bool(legacy_cumulative_flora_resume),
                    "validation_label_counts": dict(Counter(record["label"] for record in val_records)),
                    "accuracy": float(accuracy),
                },
                handle,
                indent=2,
            )

        server_state = {
            "version": 1,
            "method": method,
            "effective_method": behavior_method,
            "adapter_semantics": _adapter_semantics_for_method(behavior_method),
            "legacy_cumulative_flora_resume": bool(legacy_cumulative_flora_resume),
            "num_clients": int(num_clients),
            "num_communication_rounds": int(num_communication_rounds),
            "lora_r": int(lora_r),
            "lora_alpha": float(lora_alpha),
            "heter": bool(heter),
            "seed": int(seed),
            "completed_round": int(epoch),
            "classifier_state": _classifier_state_dict(raw_model),
            "A_cumulative": A_cumulative,
            "B_cumulative": B_cumulative,
            "B_ffa": B_ffa,
            "accuracy_list": accuracy_list,
            "expected_cumulative_residual_r": int(expected_cumulative_residual_r),
            "rng_state": _rng_state(),
        }
        if behavior_method == "flora":
            server_state["raw_model_state"] = _model_state_dict_cpu(raw_model)

        _write_latest_server_state(
            state_path,
            server_state,
        )
        print(
            f"  Saved resumable server state: {state_path}; "
            f"adapter_artifacts_retained={retain_round_artifacts}"
        )

    if end_round == num_communication_rounds:
        with open(log_path, "w") as handle:
            for accuracy in accuracy_list:
                handle.write(f"{accuracy}\n")
        _mirror_log_for_analysis(output_dir, method, num_clients)
        print(f"Final accuracies: {accuracy_list}")
        print(f"Log saved to {log_path}")
    else:
        print(f"Segment complete through round {end_round - 1}; resume from {state_path}.")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
