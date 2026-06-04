"""
Federated learning client using layercraft adapters instead of PEFT.

Drop-in replacement for client.py — same interface, same training flow,
but uses layercraft.adapter_state_dict / load_adapter_state_dict
instead of PEFT's get_peft_model_state_dict / set_peft_model_state_dict.
"""

import transformers
import os
from datasets import load_dataset
import copy
import torch
import layercraft
from .local_monitoring import LocalMetricsTrainer


def _dataloader_num_workers():
    return int(os.environ.get("FLORA_DATALOADER_NUM_WORKERS", "4"))


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, method="layercraft", local_metrics_path=None):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.method = method
        self.local_metrics_path = local_metrics_path
        self.local_monitoring_enabled = local_metrics_path is not None

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size, local_train_monitor_size=500):
        self.local_monitor_train_dataset = None
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            if self.local_monitoring_enabled:
                if local_train_monitor_size <= 0:
                    raise ValueError("local_train_monitor_size must be greater than zero.")
                local_train = local_train_val["train"].shuffle(seed=42)
                local_eval = local_train_val["test"].shuffle(seed=42)
                self.local_train_dataset = local_train.map(generate_and_tokenize_prompt)
                self.local_eval_dataset = local_eval.map(generate_and_tokenize_prompt)
                monitor_size = min(int(local_train_monitor_size), len(local_train))
                self.local_monitor_train_dataset = local_train.select(range(monitor_size)).map(
                    generate_and_tokenize_prompt
                )
            else:
                self.local_train_dataset = (
                    local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
                )
                self.local_eval_dataset = (
                    local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
                )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
            if self.local_monitoring_enabled:
                raise ValueError("Local monitoring requires local_val_set_size > 0.")
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        monitoring = self.local_monitoring_enabled and self.local_val_set_size > 0
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch" if monitoring else ("steps" if self.local_val_set_size > 0 else "no"),
            save_strategy="steps",
            eval_steps=None if monitoring else (200 if self.local_val_set_size > 0 else None),
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=False if monitoring else (True if self.local_val_set_size > 0 else False),
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            dataloader_num_workers=_dataloader_num_workers(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainer_kwargs = dict(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        if monitoring:
            self.local_trainer = LocalMetricsTrainer(
                **trainer_kwargs,
                local_monitor_train_dataset=self.local_monitor_train_dataset,
                local_metrics_path=self.local_metrics_path,
                local_method=self.method,
                local_client_id=self.client_id,
            )
        else:
            self.local_trainer = transformers.Trainer(**trainer_kwargs)

    def initiate_local_training(self):
        self.model.config.use_cache = False

        # Save a deep copy of current adapter weights (before training)
        self.params_dict_old = copy.deepcopy(layercraft.adapter_state_dict(self.model))

        # Override model.state_dict() so that Trainer only saves adapter weights
        self.model.state_dict = (
            lambda instance, *_, **__: layercraft.adapter_state_dict(instance)
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def evaluate_local_baseline(self, epoch, train_on_inputs):
        if self.local_monitoring_enabled:
            self.local_trainer.evaluate_local_baseline(epoch, train_on_inputs)

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)

        # Save trained adapter weights to disk
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        # Restore pre-training adapter weights
        layercraft.load_adapter_state_dict(self.model, self.params_dict_old)

        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
