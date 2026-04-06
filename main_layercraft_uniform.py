"""
Federated instruction-tuning — PEFT backbone + uniform layercraft adapters.

Design: identical to main.py, with one change — after get_peft_model() wraps the
model, every PEFT lora.Linear is replaced by a single layercraft adapter type.

This means:
  - All layers use the same adapter type (lora, lora_nonlinear, shim, baba, …)
  - Stacking (FLoRA) works identically — A and B matrices are stacked on the server
  - For nonlinear adapters (e.g. lora_nonlinear): σ is part of the forward pass,
    not the weights, so aggregation is unchanged
  - No per-layer config, no layer index parsing, no dispatch complexity

Usage — plain LoRA (should match main.py results):
    python main_layercraft_uniform.py \
        --global_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data_path ./data_wiz --output_dir ./lc-uniform-lora/ \
        --adapter_type lora --stacking True --seed 0

Usage — nonlinear LoRA:
    python main_layercraft_uniform.py ... --adapter_type lora_nonlinear

Usage — SHIM:
    python main_layercraft_uniform.py ... --adapter_type shim
"""

import os
import copy
from typing import List

import fire
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaTokenizer, LlamaForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel,
)
from peft import LoraConfig, get_peft_model

import layercraft
from layercraft.adapters import (
    LoRAAdapter, LoRA_SHIM_Adapter, LoRA_BABA_Adapter,
    LoRANonLinearAdapter, LoRANonLinearB,
)
from fed_utils.client_layercraft import GeneralClient
from fed_utils.model_aggregation_layercraft import FedAvg
from fed_utils.client_participation_scheduling import client_selection
from fed_utils.evaluation import global_evaluation
from utils.prompter import Prompter


# ---------------------------------------------------------------------------
# Helpers — uniform swap (no per-layer logic)
# ---------------------------------------------------------------------------

def _peft_lora_cls():
    """Return PEFT's LoRA Linear class, handling old and new PEFT versions."""
    try:
        from peft.tuners.lora.layer import Linear as Cls
    except ImportError:
        from peft.tuners.lora import Linear as Cls
    return Cls


def _build_adapter(base_linear, adapter_type, r, alpha, dropout):
    """Instantiate the layercraft adapter that wraps base_linear."""
    t = adapter_type
    if t is None or t == "lora":
        return LoRAAdapter(base_linear, r=r, alpha=alpha, dropout=dropout)
    elif t == "lora_nonlinear":
        return LoRANonLinearAdapter(base_linear, r=r, alpha=alpha, dropout=dropout)
    elif t == "lora_nonlinear_b":
        return LoRANonLinearB(base_linear, r=r, alpha=alpha, dropout=dropout)
    elif t == "shim":
        return LoRA_SHIM_Adapter(base_linear, r=r, alpha=alpha, dropout=dropout)
    elif t == "baba":
        return LoRA_BABA_Adapter(base_linear, r1=r, r2=r, alpha=alpha, dropout=dropout)
    else:
        raise ValueError(f"Unknown adapter_type: {t!r}")


def _swap_peft_to_layercraft(peft_model, adapter_type, r, alpha, dropout):
    """
    Replace every PEFT lora.Linear in peft_model with a layercraft adapter.

    PEFT's PeftModel wrapper, device hooks, and training infrastructure are
    left intact — only the inner forward computation changes.
    Returns the number of modules swapped.
    """
    PeftLoraLinear = _peft_lora_cls()
    count = 0
    for name, module in list(peft_model.named_modules()):
        if not isinstance(module, PeftLoraLinear):
            continue
        # new PEFT (>=0.4): module.base_layer; old PEFT: module IS the nn.Linear
        base_linear = getattr(module, "base_layer", module)
        adapter = _build_adapter(base_linear, adapter_type, r, alpha, dropout)
        adapter = adapter.to(base_linear.weight.device)

        # Navigate to parent and replace
        parts = name.split(".")
        parent = peft_model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], adapter)
        count += 1
    return count


def _wrap_and_swap(raw_model, r, alpha, dropout, adapter_type,
                   target_modules, global_model):
    """
    Wrap raw_model with PEFT LoRA then swap every PEFT adapter with layercraft.
    Returns (peft_model, adapter_count).
    """
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
        task_type="CAUSAL_LM", base_model_name_or_path=global_model,
    )
    peft_model = get_peft_model(raw_model, config)
    count = _swap_peft_to_layercraft(peft_model, adapter_type, r, alpha, dropout)
    return peft_model, count


def _merge_and_unwrap(peft_model):
    """
    Merge layercraft ΔW into base weights and unwrap the PeftModel.
    Returns the raw base model with merged weights (clean base for next round).
    Only valid for linear adapters that have a deltaW() method.
    """
    layercraft.merge_adapters(peft_model)
    return peft_model.base_model.model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fl_finetune(
    # model / data
    global_model: str = "huggyllama/llama-7b",
    data_path: str = "./data",
    output_dir: str = "./lc-uniform-out/",
    # FL
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1,
    num_communication_rounds: int = 5,
    num_clients: int = 10,
    # local training
    local_batch_size: int = 128,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 3,
    local_learning_rate: float = 3e-4,
    local_val_set_size: int = 0,
    local_save_steps: int = 3,
    cutoff_len: int = 512,
    # LoRA / adapter
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    adapter_type: str = "lora",   # lora | lora_nonlinear | lora_nonlinear_b | shim | baba
    # llm
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    prompt_template_name: str = "alpaca",
    # aggregation
    stacking: bool = False,
    # evaluation
    dev_data_path: str = "./mmlu_test_1444.jsonl",
    # heterogeneous
    heter: bool = False,
    local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
    zero_padding: bool = False,
    # reproducibility
    seed: int = 42,
):
    print(
        f"Federated Finetuning — PEFT backbone + uniform layercraft adapters:\n"
        f"global_model: {global_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"adapter_type: {adapter_type}\n"
        f"lora_r: {lora_r}  lora_alpha: {lora_alpha}  lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"stacking: {stacking}  heter: {heter}  zero_padding: {zero_padding}\n"
        f"num_communication_rounds: {num_communication_rounds}\n"
        f"num_clients: {num_clients}\n"
        f"seed: {seed}\n"
    )

    assert global_model, "Please specify --global_model"
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # ----------------------------------------------------------------
    # Model + tokenizer  (identical to main.py)
    # ----------------------------------------------------------------
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
            global_model, load_in_8bit=False, torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model in ("google/gemma-2b", "google/gemma-7b"):
        raw_model = AutoModelForCausalLM.from_pretrained(
            global_model, load_in_8bit=False, torch_dtype=torch.float32,
            device_map=device_map, token="your_token",
        )
    else:
        raw_model = LlamaForCausalLM.from_pretrained(
            global_model, load_in_8bit=False, torch_dtype=torch.float32,
            device_map=device_map, token="your_token",
        )

    if global_model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model in ("google/gemma-2b", "google/gemma-7b"):
        tokenizer = AutoTokenizer.from_pretrained(global_model, token="your_token")
    else:
        tokenizer = LlamaTokenizer.from_pretrained(global_model, token="your_token")

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

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
        if data_path == "./data/10":
            full_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"], data_point["response"],
            )
        elif data_path in ("./data_wiz/10", "./data_mix/20"):
            full_prompt = prompter.generate_prompt(
                data_point["instruction"], None, data_point["output"],
            )
        else:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"], data_point["output"],
            )
        tokenized = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point.get("context")
            )
            user_len = len(tokenize(user_prompt, add_eos_token=False)["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    # ----------------------------------------------------------------
    # Shared kwargs for all _wrap_and_swap calls
    # ----------------------------------------------------------------
    swap_kwargs = dict(
        dropout=lora_dropout,
        adapter_type=adapter_type,
        target_modules=lora_target_modules,
        global_model=global_model,
    )

    # ----------------------------------------------------------------
    # Non-stacking homogeneous: wrap once, all clients share the model
    # ----------------------------------------------------------------
    if not stacking and not heter:
        model, count = _wrap_and_swap(raw_model, r=lora_r, alpha=lora_alpha, **swap_kwargs)
        stats = layercraft.count_parameters(model)
        print(f"Global model: {count} adapters swapped to layercraft ({adapter_type})")
        print(f"  Trainable: {stats['trainable']:,}  Total: {stats['total']:,}")
    else:
        model = raw_model   # clean base; per-client models created in loop

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------------------------------
    # Federated training loop
    # ----------------------------------------------------------------
    print("\nThe process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    acc_list = []

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy,
            other_info=epoch,
        )

        for client_id in selected_clients_set:

            if stacking or heter:
                client_r     = local_ranks[client_id] if heter else lora_r
                client_alpha = (2 * local_ranks[client_id]) if heter else lora_alpha
                model_client, cnt = _wrap_and_swap(
                    copy.deepcopy(model), r=client_r, alpha=client_alpha, **swap_kwargs
                )
                stats = layercraft.count_parameters(model_client)
                print(f"  Client_{client_id}: {cnt} adapters (r={client_r}), "
                      f"trainable={stats['trainable']:,}")
            else:
                model_client = model

            client = GeneralClient(client_id, model_client, data_path, output_dir)

            print(f"\nPreparing the local dataset and trainer for Client_{client_id}")
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer, local_micro_batch_size, gradient_accumulation_steps,
                local_num_epochs, local_learning_rate, group_by_length, ddp,
            )

            print(f"Initiating the local training of Client_{client_id}")
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print(f"\nTerminating the local training of Client_{client_id}")
            (
                model_client,
                local_dataset_len_dict,
                previously_selected_clients_set,
                last_client_id,
            ) = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            del client
            if stacking or heter:
                del model_client

        # ---- Aggregation ----
        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(
            model, selected_clients_set, output_dir, local_dataset_len_dict,
            epoch, stacking, lora_r, heter, local_ranks, zero_padding,
        )

        # ---- Stacking: build eval model with stacked rank, then merge ----
        if stacking:
            stacked_path    = os.path.join(output_dir, str(epoch), "adapter_model.bin")
            stacked_weights = torch.load(stacked_path, map_location="cpu")

            # Infer stacked rank from first A matrix in saved weights
            stacked_r = lora_r * len(selected_clients_set)
            for key, val in stacked_weights.items():
                if key.endswith(".A") or key.endswith(".A1"):
                    stacked_r = val.shape[0]
                    break
            stacked_alpha = lora_alpha * (stacked_r // lora_r)

            # Build eval model with stacked rank — same adapter_type as clients,
            # so σ (if any) is already present in the forward pass
            model_eval, _ = _wrap_and_swap(
                copy.deepcopy(model), r=stacked_r, alpha=stacked_alpha, **swap_kwargs
            )
            layercraft.load_adapter_state_dict(model_eval, stacked_weights)
            print(f"  Stacked adapter loaded: r={stacked_r}, alpha={stacked_alpha}")

            acc = global_evaluation(model_eval, tokenizer, prompter, dev_data_path)
            print(f"Acc of Epoch {epoch} is: {acc}")
            acc_list.append(acc)

            # Merge ΔW into base weights → clean base for next round.
            # NOTE: only valid for linear adapters (lora, shim, baba).
            # For nonlinear adapters, the merged model is used as-is but the
            # adapter cannot be collapsed into a single weight matrix.
            if adapter_type in ("lora", "shim", "baba", None):
                model = _merge_and_unwrap(model_eval)
                print("  Merged stacked adapter into base weights")
            else:
                # Nonlinear: keep model_eval as the new base (with adapters intact)
                # The next round wraps it again on top, compounding adapters.
                # TODO: decide on the right cross-round strategy for nonlinear stacking.
                model = model_eval
                print(f"  Nonlinear adapter ({adapter_type}): base model not merged, "
                      f"carrying forward stacked adapter")
            del model_eval

        else:
            acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
            print(f"Acc of Epoch {epoch} is: {acc}")
            acc_list.append(acc)

        if epoch < (num_communication_rounds - 1):
            os.system(f"rm -rf {os.path.join(output_dir, str(epoch))}")

    # ---- Final log ----
    print(acc_list)
    filename = output_dir + "log.txt"
    with open(filename, "a") as f:
        for acc in acc_list:
            f.write(str(acc) + "\n")
    print("Log Saved")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
