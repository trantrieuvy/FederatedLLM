"""
Federated Freeze-A LoRA (FFA-LoRA), with optional nonlinear activation.

This implements the FFA-LoRA idea:

    y = W0 x + scale * B A_frozen x

and the nonlinear variant:

    y = W0 x + scale * B sigma(A_frozen x)

The A matrix is initialized once, shared by all clients, and frozen for all
communication rounds. Only B is trained locally and averaged on the server.
"""

import copy
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from fed_utils.client_participation_scheduling import client_selection
from fed_utils.evaluation import global_evaluation
from utils.dataset_schema import prompt_fields
from utils.prompter import Prompter


def _apply_activation(x, activation: str):
    activation = (activation or "none").lower()
    if activation in ("none", "linear", "identity"):
        return x
    if activation == "gelu":
        return F.gelu(x)
    if activation == "relu":
        return F.relu(x)
    if activation == "silu":
        return F.silu(x)
    if activation == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unknown activation: {activation!r}")


class FFALoRALayer(nn.Module):
    """
    LoRA-style residual adapter with frozen A and trainable B.

    A_frozen is a buffer, not a parameter. B is the only trainable parameter.
    """

    def __init__(
        self,
        linear: nn.Linear,
        A_frozen: torch.Tensor,
        B_initial: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.linear = linear
        self.scaling = scaling
        self.activation = (activation or "none").lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        A = A_frozen.detach().clone().to(dtype=torch.float32)
        self.register_buffer("A_frozen", A)

        if B_initial is None:
            B = torch.zeros(linear.out_features, A.shape[0], dtype=torch.float32)
        else:
            B = B_initial.detach().clone().to(dtype=torch.float32)
        self.B = nn.Parameter(B)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.linear(x)
        # Keep the adapter math in fp32, then cast the residual back to the
        # base layer dtype so bf16 base models do not hit dtype-mismatch errors.
        adapter_input = self.dropout(x).to(dtype=self.A_frozen.dtype)
        hidden = F.linear(adapter_input, self.A_frozen)
        hidden = _apply_activation(hidden, self.activation)
        update = F.linear(hidden.to(dtype=self.B.dtype), self.B)
        return y + (self.scaling * update).to(dtype=y.dtype)


def _set_submodule(model, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _target_linear_modules(model, target_modules: List[str]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            yield name, module


def _init_frozen_A(
    model,
    target_modules: List[str],
    r: int,
    seed: int,
    init_std: float,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    A_dict = {}
    for name, module in _target_linear_modules(model, target_modules):
        A_dict[name] = torch.randn(
            r,
            module.in_features,
            generator=generator,
            dtype=torch.float32,
        ) * init_std
    return A_dict


def _init_zero_B(model, target_modules: List[str], r: int) -> Dict[str, torch.Tensor]:
    B_dict = {}
    for name, module in _target_linear_modules(model, target_modules):
        B_dict[name] = torch.zeros(module.out_features, r, dtype=torch.float32)
    return B_dict


def _inject_ffa_adapters(
    base_model,
    target_modules: List[str],
    A_frozen_dict: Dict[str, torch.Tensor],
    B_dict: Dict[str, torch.Tensor],
    scaling: float,
    dropout: float,
    activation: str,
    client_r: Optional[int] = None,
):
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


def _ffa_B_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if name.endswith(".B")
    }


def _B_state_to_module_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key[:-2]: value for key, value in state_dict.items() if key.endswith(".B")}


def _module_B_to_state_dict(B_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"{name}.B": value.detach().cpu().clone() for name, value in B_dict.items()}


def _module_A_to_state_dict(A_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"{name}.A_frozen": value.detach().cpu().clone() for name, value in A_dict.items()}


def _aggregate_B_only(
    selected_clients: List[int],
    output_dir: str,
    local_dataset_len_dict: Dict[int, int],
    epoch: int,
    global_B_template: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    dataset_sizes = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients],
        dtype=torch.float32,
    )
    weights = dataset_sizes / dataset_sizes.sum()
    print("  Weights:", weights)

    aggregated = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in global_B_template.items()
    }

    for idx, client_id in enumerate(selected_clients):
        single_output_dir = os.path.join(
            output_dir,
            str(epoch),
            f"local_output_{client_id}",
            "pytorch_model.bin",
        )
        single_weights = torch.load(single_output_dir, map_location="cpu")
        single_B = _B_state_to_module_dict(single_weights)
        weight = weights[idx]

        for name, client_B in single_B.items():
            if name not in aggregated:
                raise KeyError(f"Unexpected B key from client {client_id}: {name}")
            target = aggregated[name]
            client_B = client_B.to(dtype=torch.float32)
            if client_B.shape == target.shape:
                target += client_B * weight
            elif client_B.ndim == 2 and target.ndim == 2 and client_B.shape[0] == target.shape[0]:
                if client_B.shape[1] > target.shape[1]:
                    raise ValueError(
                        f"Client B rank {client_B.shape[1]} exceeds global rank "
                        f"{target.shape[1]} for {name}"
                    )
                target[:, : client_B.shape[1]] += client_B * weight
            else:
                raise ValueError(
                    f"Cannot aggregate B for {name}: client shape {tuple(client_B.shape)} "
                    f"vs global shape {tuple(target.shape)}"
                )

    round_dir = os.path.join(output_dir, str(epoch))
    os.makedirs(round_dir, exist_ok=True)
    torch.save(_module_B_to_state_dict(aggregated), os.path.join(round_dir, "adapter_model.bin"))
    return aggregated, [float(w.item()) for w in weights]


class FFAClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{client_id}.json")
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(
            self.output_dir,
            "trainer_saved",
            f"local_output_{client_id}",
        )

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size,
                shuffle=True,
                seed=42,
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(
                generate_and_tokenize_prompt
            )
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(
        self,
        tokenizer,
        local_micro_batch_size,
        gradient_accumulation_steps,
        local_num_epochs,
        local_learning_rate,
        group_by_length,
        ddp,
    ):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            dataloader_num_workers=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.model.state_dict = (
            lambda instance, *_, **__: _ffa_B_state_dict(instance)
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(
        self,
        epoch,
        local_dataset_len_dict,
        previously_selected_clients_set,
    ):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        single_output_dir = os.path.join(
            self.output_dir,
            str(epoch),
            f"local_output_{self.client_id}",
        )
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(single_output_dir, "pytorch_model.bin"))

        previously_selected_clients_set = previously_selected_clients_set | {self.client_id}
        return self.model, local_dataset_len_dict, previously_selected_clients_set, self.client_id


def fl_finetune(
    # model / data
    global_model: str = "huggyllama/llama-7b",
    data_path: str = "./data",
    output_dir: str = "./ffa-out/",
    # FL
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1,
    num_communication_rounds: int = 5,
    num_clients: int = 10,
    # local training
    local_batch_size: int = 128,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 1,
    local_learning_rate: float = 3e-4,
    local_val_set_size: int = 0,
    cutoff_len: int = 512,
    # adapter
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    activation: str = "gelu",
    A_init_std: float = 0.02,
    heter: bool = False,
    local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
    # llm
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    prompt_template_name: str = "alpaca",
    # evaluation
    dev_data_path: str = "./mmlu_test_1444.jsonl",
    # reproducibility
    seed: int = 42,
):
    activation = (activation or "none").lower()
    if heter and len(local_ranks) < num_clients:
        raise ValueError("local_ranks must provide one rank per client when heter=True")

    base_scaling = lora_alpha / lora_r
    global_r = max(local_ranks[:num_clients]) if heter else lora_r

    print(
        "Federated Freeze-A LoRA\n"
        f"  global_model:             {global_model}\n"
        f"  num_communication_rounds: {num_communication_rounds}\n"
        f"  num_clients:              {num_clients}\n"
        f"  activation:               {activation}\n"
        f"  lora_r: {lora_r}  global_r: {global_r}  lora_alpha: {lora_alpha}\n"
        f"  effective scaling:        {base_scaling}\n"
        f"  trainable side:           B only\n"
        f"  A_init_std:               {A_init_std}\n"
        f"  local_num_epochs:         {local_num_epochs}\n"
        f"  local_learning_rate:      {local_learning_rate}\n"
        f"  seed:                     {seed}\n"
    )

    assert global_model
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), (
        f"Data path {data_path} not found. Run client_data_allocation.py first."
    )

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = "auto"
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if global_model == "gpt2":
        raw_model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model in ("google/gemma-2b", "google/gemma-7b"):
        raw_model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token="your_token",
        )
        tokenizer = AutoTokenizer.from_pretrained(global_model, token="your_token")
    else:
        raw_model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token="your_token",
        )
        tokenizer = LlamaTokenizer.from_pretrained(global_model, token="your_token")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    for p in raw_model.parameters():
        p.requires_grad = False

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        instruction, prompt_input, prompt_label, _ = prompt_fields(data_point)
        full_prompt = prompter.generate_prompt(
            instruction,
            prompt_input,
            prompt_label,
        )
        tokenized = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction,
                prompt_input,
            )
            user_len = len(tokenize(user_prompt, add_eos_token=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    raw_model.to("cpu")
    A_frozen_dict = _init_frozen_A(
        raw_model,
        target_modules=lora_target_modules,
        r=global_r,
        seed=seed,
        init_std=A_init_std,
    )
    B_global_dict = _init_zero_B(
        raw_model,
        target_modules=lora_target_modules,
        r=global_r,
    )

    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(_module_A_to_state_dict(A_frozen_dict), os.path.join(output_dir, "A_frozen.bin"))
    with open(os.path.join(output_dir, "ffa_config.json"), "w") as f:
        json.dump(
            dict(
                global_model=global_model,
                num_clients=int(num_clients),
                lora_r=int(lora_r),
                global_r=int(global_r),
                lora_alpha=float(lora_alpha),
                effective_scaling=float(base_scaling),
                lora_target_modules=list(lora_target_modules),
                activation=activation,
                A_init_std=float(A_init_std),
                A_init_seed=int(seed),
                heter=bool(heter),
                local_ranks=[int(r) for r in local_ranks[:num_clients]] if heter else None,
            ),
            f,
            indent=2,
        )

    acc_list = []
    print("\nStarting FFA-LoRA federated training.")
    for epoch in tqdm(range(num_communication_rounds)):
        print(f"\n=== Round {epoch} ===")
        selected_clients = sorted(
            client_selection(
                num_clients,
                client_selection_frac,
                client_selection_strategy,
                other_info=epoch,
            )
        )
        print(f"  Selected clients: {selected_clients}")

        local_dataset_len_dict = {}
        previously_selected_clients_set = set()

        for client_id in selected_clients:
            client_r = local_ranks[client_id] if heter else global_r
            print(f"  Client_{client_id}: rank={client_r}")
            raw_model.to("cpu")
            model_client, n_adapters = _inject_ffa_adapters(
                copy.deepcopy(raw_model),
                target_modules=lora_target_modules,
                A_frozen_dict=A_frozen_dict,
                B_dict=B_global_dict,
                scaling=base_scaling,
                dropout=lora_dropout,
                activation=activation,
                client_r=client_r if heter else None,
            )
            model_client.to("cuda")
            trainable = sum(p.numel() for p in model_client.parameters() if p.requires_grad)
            print(f"    adapters={n_adapters}, trainable_B_params={trainable:,}")

            client = FFAClient(client_id, model_client, data_path, output_dir)
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp,
            )
            client.initiate_local_training()
            client.train()
            (
                model_client,
                local_dataset_len_dict,
                previously_selected_clients_set,
                _,
            ) = client.terminate_local_training(
                epoch,
                local_dataset_len_dict,
                previously_selected_clients_set,
            )
            del client, model_client
            torch.cuda.empty_cache()

        print("  Aggregating B only.")
        B_global_dict, aggregation_weights = _aggregate_B_only(
            selected_clients,
            output_dir,
            local_dataset_len_dict,
            epoch,
            B_global_dict,
        )

        model_eval, _ = _inject_ffa_adapters(
            copy.deepcopy(raw_model),
            target_modules=lora_target_modules,
            A_frozen_dict=A_frozen_dict,
            B_dict=B_global_dict,
            scaling=base_scaling,
            dropout=0.0,
            activation=activation,
            client_r=None,
        )
        model_eval.to("cuda")
        acc = global_evaluation(model_eval, tokenizer, prompter, dev_data_path)
        print(f"  Acc round {epoch}: {acc}")
        acc_list.append(acc)
        del model_eval
        torch.cuda.empty_cache()

        epoch_dir = os.path.join(output_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        with open(os.path.join(epoch_dir, "round_config.json"), "w") as f:
            json.dump(
                dict(
                    epoch=int(epoch),
                    activation=activation,
                    lora_r=int(lora_r),
                    global_r=int(global_r),
                    lora_alpha=float(lora_alpha),
                    effective_scaling=float(base_scaling),
                    heter=bool(heter),
                    selected_clients=[int(c) for c in selected_clients],
                    aggregation_weights=aggregation_weights,
                    local_dataset_sizes={
                        int(c): int(local_dataset_len_dict[c]) for c in selected_clients
                    },
                    per_client_ranks={
                        int(c): int(local_ranks[c]) for c in selected_clients
                    }
                    if heter
                    else None,
                    trainable_B_params=int(sum(v.numel() for v in B_global_dict.values())),
                    frozen_A_params=int(sum(v.numel() for v in A_frozen_dict.values())),
                ),
                f,
                indent=2,
            )

    print(f"\nFinal accuracies: {acc_list}")
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        for acc in acc_list:
            f.write(str(acc) + "\n")
    print("Log saved.")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
