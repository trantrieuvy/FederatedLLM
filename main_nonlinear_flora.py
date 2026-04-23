"""
Multi-round nonlinear FLoRA.

Each round clients train a residual nonlinear adapter  B·σ(A·x)  on top of
the stacked frozen adapter from the previous round.  After N rounds:

    y = W0·x  +  B0·σ(A0·x)  +  B1·σ(A1·x)  +  …  +  BN·σ(AN·x)

Changes vs. linear FLoRA (main.py):
  1. Forward pass: B·σ(A·x) instead of B@A @ x
  2. After each round: stacked (A, B) are frozen and distributed to clients.
     Clients train a fresh residual adapter on top each round instead of
     merging ΔW into W0 (which is impossible for nonlinear adapters).
"""

import os
import copy
import json
from typing import List

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaTokenizer, LlamaForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel,
)

from fed_utils.client_layercraft import GeneralClient
from fed_utils.model_aggregation_layercraft import FedAvg
from fed_utils.client_participation_scheduling import client_selection
from fed_utils.evaluation import global_evaluation
from utils.prompter import Prompter
from utils.dataset_schema import prompt_fields


# ---------------------------------------------------------------------------
# Nonlinear LoRA layer
# ---------------------------------------------------------------------------

class NonlinearLoRALayer(nn.Module):
    """
    y = W0·x
      + frozen_scaling · B_frozen · σ(A_frozen · x)   ← stacked from prev round (frozen)
      + new_scaling    · B_new    · σ(A_new    · x)   ← this round (trainable)

    Round 0: A_frozen / B_frozen are None, only the new adapter contributes.
    """

    def __init__(self, linear, r, alpha, dropout=0.0, init_std=0.02,
                A_frozen=None, B_frozen=None, frozen_scaling=None):
        if A_frozen is not None and frozen_scaling is None:
            raise ValueError("frozen_scaling must be provided when A_frozen is set")

        super().__init__()
        self.linear = linear
        self.new_scaling = alpha / r

        self.A_new = nn.Parameter(torch.empty(r, linear.in_features))
        nn.init.normal_(self.A_new, std=init_std)
        self.B_new = nn.Parameter(torch.zeros(linear.out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if A_frozen is not None:
            self.register_buffer("A_frozen", A_frozen.detach().clone())
            self.register_buffer("B_frozen", B_frozen.detach().clone())
            self.frozen_scaling = frozen_scaling
        else:
            self.A_frozen = None
            self.B_frozen = None

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.linear(x)
        if self.A_frozen is not None:
            y = y + self.frozen_scaling * F.linear(
                F.gelu(F.linear(x, self.A_frozen)), self.B_frozen
            )
        return y + self.new_scaling * F.linear(
            F.gelu(F.linear(self.dropout(x), self.A_new)), self.B_new
        )


# ---------------------------------------------------------------------------
# Adapter injection helpers
# ---------------------------------------------------------------------------

def _set_submodule(model, dotted_name, new_module):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _inject_adapters(base_model, target_modules, r, alpha, dropout,
                     A_frozen_dict, B_frozen_dict, frozen_scaling):
    """
    Replace every target nn.Linear with NonlinearLoRALayer.
    A_frozen_dict / B_frozen_dict: {module_name → tensor} from the previous round
    (both None for round 0).
    """
    count = 0
    for name, module in list(base_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        A_f = A_frozen_dict.get(name) if A_frozen_dict else None
        B_f = B_frozen_dict.get(name) if B_frozen_dict else None
        adapter = NonlinearLoRALayer(
            module, r=r, alpha=alpha, dropout=dropout,
            A_frozen=A_f, B_frozen=B_f, frozen_scaling=frozen_scaling,
        ).to(module.weight.device)
        _set_submodule(base_model, name, adapter)
        count += 1
    return base_model, count


def _parse_stacked(stacked_weights):
    """
    Split stacked state dict into {module_name → A} and {module_name → B}.
    Keys look like  "…q_proj.A_new" / "…q_proj.B_new".
    """
    A_dict, B_dict = {}, {}
    for key, val in stacked_weights.items():
        if key.endswith(".A_new"):
            A_dict[key[:-6]] = val
        elif key.endswith(".B_new"):
            B_dict[key[:-6]] = val
    return A_dict, B_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fl_finetune(
    # model / data
    global_model: str = "huggyllama/llama-7b",
    data_path: str = "./data",
    output_dir: str = "./nonlinear-flora-out/",
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
    print(
        f"Multi-round Nonlinear FLoRA\n"
        f"  global_model:             {global_model}\n"
        f"  num_communication_rounds: {num_communication_rounds}\n"
        f"  num_clients:              {num_clients}\n"
        f"  lora_r: {lora_r}  lora_alpha: {lora_alpha}\n"
        f"  local_num_epochs:         {local_num_epochs}\n"
        f"  local_learning_rate:      {local_learning_rate}\n"
        f"  seed:                     {seed}\n"
    )

    assert global_model
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), \
        f"Data path {data_path} not found — run client_data_allocation.py first"

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = "auto"
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # ---- Load base model (stays frozen throughout all rounds) ----
    if global_model == "gpt2":
        raw_model = GPT2LMHeadModel.from_pretrained(
            global_model, load_in_8bit=False, torch_dtype=torch.float32,
            device_map=device_map,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model in ("google/gemma-2b", "google/gemma-7b"):
        raw_model = AutoModelForCausalLM.from_pretrained(
            global_model, load_in_8bit=False, torch_dtype=torch.float32,
            device_map=device_map, token="your_token",
        )
        tokenizer = AutoTokenizer.from_pretrained(global_model, token="your_token")
    else:
        raw_model = LlamaForCausalLM.from_pretrained(
            global_model, load_in_8bit=False, torch_dtype=torch.bfloat16,
            device_map=device_map, token="your_token",
        )
        tokenizer = LlamaTokenizer.from_pretrained(global_model, token="your_token")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Freeze all base model parameters before injection
    for p in raw_model.parameters():
        p.requires_grad = False

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, truncation=True, max_length=cutoff_len,
            padding=False, return_tensors=None,
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
            instruction, prompt_input, prompt_label,
        )
        tokenized = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction, prompt_input,
            )
            user_len = len(tokenize(user_prompt, add_eos_token=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- State shared across rounds ----
    # None until after round 0; then maps module_name → stacked tensor.
    A_frozen_dict, B_frozen_dict = None, None
    # scaling = alpha/r (same formula as each client's per-adapter scaling)
    frozen_scaling = lora_alpha / lora_r

    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)
    acc_list = []

    print("\nStarting multi-round nonlinear FLoRA …")
    for epoch in tqdm(range(num_communication_rounds)):
        print(f"\n=== Round {epoch} ===")
        if A_frozen_dict is not None:
            stacked_r = next(iter(A_frozen_dict.values())).shape[0]
            n_frozen = (
                sum(v.numel() for v in A_frozen_dict.values())
                + sum(v.numel() for v in B_frozen_dict.values())
            )
            print(f"  Frozen adapter: stacked_r={stacked_r}, params={n_frozen:,}")

        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy,
            other_info=epoch,
        )
        print(f"  Selected clients: {sorted(selected_clients_set)}")

        local_dataset_len_dict = {}
        previously_selected_clients_set = set()

        for client_id in selected_clients_set:
            # Build per-client model: frozen base + fresh trainable adapter
            client_r = local_ranks[client_id] if heter else lora_r
            # Keep alpha/r constant across clients so frozen_scaling is valid
            # for all blocks after stacking (e.g. r=64 → alpha=128, r=4 → alpha=8,
            # all giving alpha/r = frozen_scaling = lora_alpha/lora_r).
            client_alpha = frozen_scaling * client_r
            raw_model.to('cpu')
            model_client, n_adapters = _inject_adapters(
                copy.deepcopy(raw_model),
                target_modules=lora_target_modules,
                r=client_r, alpha=client_alpha, dropout=lora_dropout,
                A_frozen_dict=A_frozen_dict,
                B_frozen_dict=B_frozen_dict,
                frozen_scaling=frozen_scaling,
            )
            model_client.to('cuda')
            trainable = sum(p.numel() for p in model_client.parameters() if p.requires_grad)
            print(f"  Client_{client_id}: {n_adapters} adapters, trainable={trainable:,}")

            client = GeneralClient(client_id, model_client, data_path, output_dir)
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer, local_micro_batch_size, gradient_accumulation_steps,
                local_num_epochs, local_learning_rate, group_by_length, ddp,
            )
            client.initiate_local_training()
            client.train()
            (model_client, local_dataset_len_dict,
             previously_selected_clients_set, _) = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set,
            )
            del client, model_client
            torch.cuda.empty_cache()

        # ---- Stack A_new / B_new across clients (reuses layercraft FedAvg) ----
        print("  Stacking …")
        FedAvg(
            raw_model, selected_clients_set, output_dir, local_dataset_len_dict,
            epoch, stacking=True, lora_r=lora_r, heter=heter,
            local_ranks=local_ranks if heter else [], zero_padding=False,
            nonlinear=True,
        )
        stacked_path = os.path.join(output_dir, str(epoch), "adapter_model.bin")
        A_frozen_dict, B_frozen_dict = _parse_stacked(
            torch.load(stacked_path, map_location="cpu")
        )
        stacked_r = next(iter(A_frozen_dict.values())).shape[0]
        if heter:
            client_ranks_str = "+".join(str(local_ranks[c]) for c in sorted(selected_clients_set))
            print(f"  Stacked adapter: r={stacked_r} ({client_ranks_str})")
        else:
            print(f"  Stacked adapter: r={stacked_r} ({len(selected_clients_set)} clients × {lora_r})")

        # ---- Evaluate: frozen adapters only, B_new=zeros → no leakage ----
        model_eval, _ = _inject_adapters(
            copy.deepcopy(raw_model),
            target_modules=lora_target_modules,
            r=lora_r, alpha=lora_alpha, dropout=0.0,
            A_frozen_dict=A_frozen_dict,
            B_frozen_dict=B_frozen_dict,
            frozen_scaling=frozen_scaling,
        )
        model_eval.to('cuda')
        acc = global_evaluation(model_eval, tokenizer, prompter, dev_data_path)
        print(f"  Acc round {epoch}: {acc}")
        acc_list.append(acc)
        del model_eval
        torch.cuda.empty_cache()

        # Save round metadata
        epoch_dir = os.path.join(output_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        with open(os.path.join(epoch_dir, "round_config.json"), "w") as f:
            json.dump(dict(
                epoch=int(epoch),
                lora_r=int(lora_r), lora_alpha=int(lora_alpha), stacked_r=int(stacked_r),
                effective_scaling=frozen_scaling,
                heter=heter,
                per_client_ranks={int(c): int(local_ranks[c]) for c in sorted(selected_clients_set)} if heter else None,
                per_client_alphas={int(c): frozen_scaling * local_ranks[c] for c in sorted(selected_clients_set)} if heter else None,
                selected_clients=[int(c) for c in sorted(selected_clients_set)],
                frozen_adapter_params=int(
                    sum(v.numel() for v in A_frozen_dict.values())
                    + sum(v.numel() for v in B_frozen_dict.values())
                ),
            ), f, indent=2)

        if epoch < (num_communication_rounds - 1):
            os.system(f"rm -rf {os.path.join(output_dir, str(epoch))}")

    print(f"\nFinal accuracies: {acc_list}")
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        for acc in acc_list:
            f.write(str(acc) + "\n")
    print("Log saved.")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
