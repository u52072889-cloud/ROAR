#!/usr/bin/env python
"""Train a LoRA DPO adapter for node-summary generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from data_utils import (  # noqa: E402
    effective_model_max_length,
    resolve_local_hf_model_path,
    set_offline_hf_env,
    strict_encode,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def require_nonempty_tokenizer(tokenizer: Any, *, context: str) -> None:
    encoded = tokenizer("tokenizer sanity check", add_special_tokens=True, padding=False, truncation=False)
    if not encoded.get("input_ids"):
        raise ValueError(f"{context} tokenizer produced zero tokens. The cached tokenizer files are incomplete.")


def model_load_source(model_name_or_path: str, resolved_path: str, *, allow_download: bool) -> str:
    if Path(model_name_or_path).exists():
        return model_name_or_path
    if Path(resolved_path).exists():
        return resolved_path
    if allow_download:
        return model_name_or_path
    return resolved_path


def load_tokenizer_and_model(
    *,
    model_name_or_path: str,
    device: str,
    cache_dir: str,
    allow_download: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
) -> tuple[Any, Any]:
    model_path = resolve_local_hf_model_path(model_name_or_path, cache_dir=cache_dir)
    model_is_local = Path(model_path).exists()
    if not model_is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{model_name_or_path}' was not found in cache_dir='{cache_dir}' while offline mode is enabled. "
            "Pre-download the model into the cache or rerun with --allow-download."
        )
    source = model_load_source(model_name_or_path, str(model_path), allow_download=allow_download)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        use_fast=True,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token.")
    require_nonempty_tokenizer(tokenizer, context=f"Model '{model_name_or_path}'")

    model_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=not allow_download,
        **model_kwargs,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_target_modules),
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return tokenizer, model


def append_eos_if_missing(token_ids: List[int], eos_token_id: int | None) -> List[int]:
    output = list(token_ids)
    if eos_token_id is None:
        return output
    if not output or int(output[-1]) != int(eos_token_id):
        output.append(int(eos_token_id))
    return output


@dataclass
class PreferenceRow:
    record_id: str
    prompt_ids: List[int]
    chosen_ids: List[int]
    rejected_ids: List[int]


class PreferenceDataset(Dataset):
    def __init__(
        self,
        *,
        path: str,
        tokenizer: Any,
        model_config: Any,
        max_records: int,
    ) -> None:
        self.rows: List[PreferenceRow] = []
        context_window = effective_model_max_length(tokenizer, model_config=model_config)
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for DPO training.")
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                payload = json.loads(raw_line)
                record_id = str(payload.get("id") or "")
                prompt = str(payload.get("prompt") or "")
                chosen = str(payload.get("chosen") or "")
                rejected = str(payload.get("rejected") or "")
                if not prompt:
                    raise ValueError(f"DPO row {record_id} is missing prompt.")
                if not chosen:
                    raise ValueError(f"DPO row {record_id} is missing chosen.")
                if not rejected:
                    raise ValueError(f"DPO row {record_id} is missing rejected.")
                prompt_ids = strict_encode(
                    tokenizer,
                    prompt,
                    add_special_tokens=False,
                    context=f"DPO prompt for {record_id}",
                )
                chosen_ids = append_eos_if_missing(
                    strict_encode(
                        tokenizer,
                        chosen,
                        add_special_tokens=False,
                        context=f"DPO chosen for {record_id}",
                    ),
                    eos_token_id,
                )
                rejected_ids = append_eos_if_missing(
                    strict_encode(
                        tokenizer,
                        rejected,
                        add_special_tokens=False,
                        context=f"DPO rejected for {record_id}",
                    ),
                    eos_token_id,
                )
                if context_window > 0:
                    chosen_total = len(prompt_ids) + len(chosen_ids)
                    rejected_total = len(prompt_ids) + len(rejected_ids)
                    if chosen_total > context_window:
                        raise ValueError(
                            f"DPO chosen example {record_id} requires {chosen_total} tokens, "
                            f"exceeding model limit {context_window}. No truncation is allowed."
                        )
                    if rejected_total > context_window:
                        raise ValueError(
                            f"DPO rejected example {record_id} requires {rejected_total} tokens, "
                            f"exceeding model limit {context_window}. No truncation is allowed."
                        )
                self.rows.append(
                    PreferenceRow(
                        record_id=record_id,
                        prompt_ids=prompt_ids,
                        chosen_ids=chosen_ids,
                        rejected_ids=rejected_ids,
                    )
                )
                if max_records and len(self.rows) >= max_records:
                    break
        if not self.rows:
            raise ValueError(f"No DPO rows loaded from {path}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {
            "id": row.record_id,
            "prompt_ids": list(row.prompt_ids),
            "chosen_ids": list(row.chosen_ids),
            "rejected_ids": list(row.rejected_ids),
        }


class PreferenceCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def _pack(self, prompt_ids: List[int], response_ids: List[int]) -> Dict[str, List[int]]:
        input_ids = list(prompt_ids) + list(response_ids)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _pad(self, sequences: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in sequences)
        padded_input_ids: List[List[int]] = []
        padded_attention_mask: List[List[int]] = []
        padded_labels: List[List[int]] = []
        for item in sequences:
            pad_len = max_len - len(item["input_ids"])
            padded_input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            padded_labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen = [self._pack(item["prompt_ids"], item["chosen_ids"]) for item in features]
        rejected = [self._pack(item["prompt_ids"], item["rejected_ids"]) for item in features]
        batch = {}
        for prefix, sequences in (("chosen", chosen), ("rejected", rejected)):
            packed = self._pad(sequences)
            for key, value in packed.items():
                batch[f"{prefix}_{key}"] = value
        return batch


def sequence_logps(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


class DPOTrainer(Trainer):
    def __init__(self, *args: Any, beta: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        if not hasattr(self.model, "disable_adapter"):
            raise ValueError("This DPO trainer requires a PEFT model with disable_adapter().")

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del num_items_in_batch
        chosen_logps = sequence_logps(
            model=model,
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"],
        )
        rejected_logps = sequence_logps(
            model=model,
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"],
        )
        with torch.no_grad():
            with model.disable_adapter():
                ref_chosen_logps = sequence_logps(
                    model=model,
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"],
                    labels=inputs["chosen_labels"],
                )
                ref_rejected_logps = sequence_logps(
                    model=model,
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"],
                    labels=inputs["rejected_labels"],
                )
        preference_logits = self.beta * (
            (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        )
        loss = -F.logsigmoid(preference_logits).mean()
        if not return_outputs:
            return loss
        return loss, {
            "chosen_logps": chosen_logps.detach(),
            "rejected_logps": rejected_logps.detach(),
            "ref_chosen_logps": ref_chosen_logps.detach(),
            "ref_rejected_logps": ref_rejected_logps.detach(),
            "preference_logits": preference_logits.detach(),
        }


def parse_target_modules(value: str) -> List[str]:
    modules = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora-target-modules must not be empty.")
    return modules


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    args = parser.parse_args()

    if float(args.epochs) <= 0.0:
        raise ValueError("--epochs must be positive.")
    if int(args.batch) <= 0:
        raise ValueError("--batch must be positive.")
    if int(args.grad_accum) <= 0:
        raise ValueError("--grad-accum must be positive.")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive.")
    if float(args.beta) <= 0.0:
        raise ValueError("--beta must be positive.")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative.")
    if int(args.lora_r) <= 0:
        raise ValueError("--lora-r must be positive.")
    if int(args.lora_alpha) <= 0:
        raise ValueError("--lora-alpha must be positive.")
    if float(args.lora_dropout) < 0.0 or float(args.lora_dropout) >= 1.0:
        raise ValueError("--lora-dropout must be within [0, 1).")

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()

    set_seed(int(args.seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_modules = parse_target_modules(args.lora_target_modules)

    tokenizer, model = load_tokenizer_and_model(
        model_name_or_path=args.model,
        device=device,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=target_modules,
    )
    dataset = PreferenceDataset(
        path=args.data,
        tokenizer=tokenizer,
        model_config=model.config,
        max_records=int(args.max_records),
    )
    collator = PreferenceCollator(pad_token_id=int(tokenizer.pad_token_id))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_tmp"),
        do_train=True,
        do_eval=False,
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        seed=int(args.seed),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        beta=float(args.beta),
    )
    train_result = trainer.train()
    trainer.save_state()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_json(out_dir / "train_metrics.json", dict(train_result.metrics or {}))
    save_json(
        out_dir / "train_config.json",
        {
            "data": str(args.data),
            "model": str(args.model),
            "epochs": float(args.epochs),
            "batch": int(args.batch),
            "grad_accum": int(args.grad_accum),
            "lr": float(args.lr),
            "beta": float(args.beta),
            "seed": int(args.seed),
            "max_records": int(args.max_records),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target_modules": list(target_modules),
            "train_examples": len(dataset),
        },
    )


if __name__ == "__main__":
    main()
