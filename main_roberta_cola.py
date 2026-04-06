"""
Federated LoRA fine-tuning of RoBERTa-base on CoLA (GLUE benchmark).

Adapts the FederatedLLM framework (originally for CausalLM) to support
sequence classification with RoBERTa-base on the CoLA task.

Usage:
    python main_roberta_cola.py \
        --num_clients 10 \
        --num_communication_rounds 100 \
        --local_num_epochs 1 \
        --lora_r 8 \
        --lora_alpha 16

See run_cola_federated.sh for full examples.
"""

import os
from typing import List
from tqdm import tqdm
import fire
import torch
import numpy as np
import random
import copy
import json

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel,
)
from datasets import load_dataset, Dataset
from collections import OrderedDict

from fed_utils import FedAvg, client_selection
from fed_utils.evaluation_cola import global_evaluation_cola


class CoLAClient:
    """Client for federated CoLA training with RoBERTa."""

    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", f"local_output_{self.client_id}")

        with open(self.local_data_path, "r") as f:
            records = json.load(f)
        self.local_data = Dataset.from_dict({
            "sentence": [r["sentence"] for r in records],
            "label": [r["label"] for r in records],
        })

    def prepare_local_dataset(self, tokenizer, max_seq_length):
        """Tokenize local dataset for sequence classification."""
        def tokenize_fn(examples):
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )

        self.local_train_dataset = self.local_data.shuffle(seed=42).map(
            tokenize_fn, batched=True, remove_columns=["sentence"]
        )
        self.local_train_dataset.set_format("torch")

    def build_local_trainer(self, tokenizer, local_micro_batch_size, gradient_accumulation_steps,
                            local_num_epochs, local_learning_rate, ddp):
        self.train_args = TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            dataloader_drop_last=False,
            report_to="none",
        )
        self.local_trainer = Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            args=self.train_args,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict(
                (name, param.detach())
                for name, param in self.model.named_parameters()
                if "default" in name
            )
        )
        self.params_dict_new = OrderedDict(
            (name, param.detach())
            for name, param in self.model.named_parameters()
            if "default" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, os.path.join(single_output_dir, "pytorch_model.bin"))

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | {self.client_id}
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id


def fl_finetune(
    # model params
    global_model: str = "roberta-base",
    output_dir: str = "./fedlora-roberta-cola/",
    # data params
    data_path: str = "./data_cola",
    max_seq_length: int = 128,
    # FL hyperparams
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1.0,
    num_communication_rounds: int = 100,
    num_clients: int = 10,
    # Local training hyperparams
    local_batch_size: int = 32,
    local_micro_batch_size: int = 16,
    local_num_epochs: int = 1,
    local_learning_rate: float = 2e-4,
    # LoRA hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["query", "value"],
    # aggregation
    stacking: bool = False,
    # heterogeneous
    heter: bool = False,
    local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
    zero_padding: bool = False,
    # misc
    seed: int = 42,
):
    print(
        f"Federated LoRA Fine-Tuning RoBERTa-base on CoLA\n"
        f"{'='*50}\n"
        f"global_model: {global_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"num_clients: {num_clients}\n"
        f"num_communication_rounds: {num_communication_rounds}\n"
        f"client_selection_frac: {client_selection_frac}\n"
        f"local_batch_size: {local_batch_size}\n"
        f"local_micro_batch_size: {local_micro_batch_size}\n"
        f"local_num_epochs: {local_num_epochs}\n"
        f"local_learning_rate: {local_learning_rate}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"stacking: {stacking}\n"
        f"seed: {seed}\n"
    )

    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), (
        f"Data path {data_path} does not exist. "
        f"Run client_data_allocation_cola.py first."
    )

    val_data_path = os.path.join(data_path, "global_val.json")
    assert os.path.exists(val_data_path), f"Validation data not found at {val_data_path}"

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load RoBERTa-base for sequence classification
    config = AutoConfig.from_pretrained(global_model, num_labels=2, finetuning_task="cola")
    model = AutoModelForSequenceClassification.from_pretrained(
        global_model,
        config=config,
        torch_dtype=torch.float32,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(global_model)

    # Apply LoRA
    if stacking:
        lora_config = LoraConfig(
            r=lora_r * num_clients,
            lora_alpha=lora_alpha * num_clients,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("\nStarting federated fine-tuning on CoLA...")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    mcc_list = []

    for epoch in tqdm(range(num_communication_rounds), desc="Communication Rounds"):
        print(f"\n--- Communication Round {epoch} ---")

        # Client selection
        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy, other_info=epoch
        )
        print(f"Selected clients: {selected_clients_set}")

        for client_id in selected_clients_set:
            if stacking:
                client_lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                )
                model_client = copy.deepcopy(model)
                model_client = get_peft_model(model_client, client_lora_config)
            elif heter:
                client_lora_config = LoraConfig(
                    r=local_ranks[client_id],
                    lora_alpha=2 * local_ranks[client_id],
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                )
                model_client = copy.deepcopy(model)
                model_client = get_peft_model(model_client, client_lora_config)
            else:
                model_client = model

            client = CoLAClient(client_id, model_client, data_path, output_dir)

            print(f"  Preparing Client_{client_id}...")
            client.prepare_local_dataset(tokenizer, max_seq_length)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                ddp,
            )

            print(f"  Training Client_{client_id}...")
            client.initiate_local_training()
            client.train()

            print(f"  Terminating Client_{client_id}...")
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = (
                client.terminate_local_training(epoch, local_dataset_len_dict, previously_selected_clients_set)
            )
            del client

        # Aggregation
        print("  Aggregating client weights (FedAvg)...")
        model = FedAvg(
            model,
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            epoch,
            stacking,
            lora_r,
            heter,
            local_ranks,
            zero_padding,
            full=False,
        )

        # Save checkpoint
        if stacking:
            stacking_config = LoraConfig(
                r=lora_r * num_clients,
                lora_alpha=lora_alpha * num_clients,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            stacking_config.save_pretrained(os.path.join(output_dir, str(epoch)))
            model = PeftModel.from_pretrained(model, os.path.join(output_dir, str(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
            lora_config.save_pretrained(os.path.join(output_dir, str(epoch)))

        # Global evaluation
        mcc = global_evaluation_cola(model, tokenizer, val_data_path, max_seq_length)
        print(f"  MCC at round {epoch}: {mcc:.4f}")
        mcc_list.append(mcc)

        if stacking:
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(output_dir, str(epoch), "final"))

        # Clean up intermediate checkpoints
        if epoch < (num_communication_rounds - 1):
            rm_dir = os.path.join(output_dir, str(epoch))
            os.system(f"rm -rf {rm_dir}")

    # Final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"MCC per round: {mcc_list}")
    print(f"Best MCC: {max(mcc_list):.4f} at round {mcc_list.index(max(mcc_list))}")

    # Save log
    log_path = os.path.join(output_dir, "mcc_log.txt")
    with open(log_path, "w") as f:
        for i, mcc in enumerate(mcc_list):
            f.write(f"Round {i}: {mcc:.4f}\n")
        f.write(f"\nBest MCC: {max(mcc_list):.4f} at round {mcc_list.index(max(mcc_list))}\n")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
