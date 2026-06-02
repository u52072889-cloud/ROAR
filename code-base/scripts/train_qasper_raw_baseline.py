#!/usr/bin/env python
"""Train a raw linear baseline on QASPER records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from data_utils import (
    QUERY_REGION_ID,
    SPECIAL_TOKENS,
    TARGET_REGION_ID,
    effective_model_max_length,
    resolve_local_hf_model_path,
    set_offline_hf_env,
    strict_encode,
    strict_special_token_id,
)
from qasper_eval import aggregate_prediction_metrics, best_answer_scores, best_evidence_scores, qasper_question_key


class RawQasperDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Any,
        *,
        max_len: int,
        max_records: int = 0,
        filter_include_target: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.filter_include_target = bool(filter_include_target)
        self.source_path = str(path)
        loaded_records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                loaded_records.append(json.loads(line))
                if max_records and len(loaded_records) >= max_records:
                    break
        self.loaded_record_count = len(loaded_records)
        self.dropped_over_limit_count = 0
        self.dropped_over_limit_examples: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        for record in loaded_records:
            try:
                self.build_example(record, include_target=self.filter_include_target)
            except ValueError as exc:
                if self.max_len > 0 and "exceeding max_len=" in str(exc):
                    self.dropped_over_limit_count += 1
                    if len(self.dropped_over_limit_examples) < 5:
                        self.dropped_over_limit_examples.append(
                            {
                                "id": str(record.get("id") or ""),
                                "error": str(exc),
                            }
                        )
                    continue
                raise
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def drop_report(self) -> Dict[str, Any]:
        return {
            "path": self.source_path,
            "loaded_records": int(self.loaded_record_count),
            "kept_records": int(len(self.records)),
            "dropped_over_limit": int(self.dropped_over_limit_count),
            "max_len": int(self.max_len),
            "filter_include_target": bool(self.filter_include_target),
            "dropped_examples": list(self.dropped_over_limit_examples),
        }

    def _encode(self, text: str, *, context: str) -> List[int]:
        return strict_encode(self.tokenizer, text, add_special_tokens=False, context=context)

    def _render_prefix(self, record: Dict[str, Any]) -> str:
        prefix_lines: List[str] = []
        title = str(record.get("doc_meta", {}).get("title") or "").strip()
        abstract = str(record.get("doc_meta", {}).get("abstract") or "").strip()
        if title:
            prefix_lines.append(f"Title: {title}")
        if abstract:
            prefix_lines.append(f"Abstract: {abstract}")
        return "<prefix>" + "\n\n".join(prefix_lines).strip() + "</prefix>"

    def _render_query(self, record: Dict[str, Any]) -> str:
        return "<query>" + str(record.get("query", {}).get("question") or "") + "</query>"

    def _render_segment(self, segment: Dict[str, Any]) -> str:
        text = str(segment.get("text") or "")
        return "<evidence>" + text + "</evidence>"

    def _paragraph_node_ids(self, record: Dict[str, Any]) -> List[int]:
        node_ids = [
            int(node["node_id"])
            for node in (record.get("nodes") or [])
            if str(node.get("node_type") or "") == "paragraph"
        ]
        if not node_ids:
            raise ValueError(
                f"Raw baseline record {record.get('id')} is missing paragraph nodes. "
                "The prepare artifact is incomplete."
            )
        return node_ids

    def _raw_document_text(self, record: Dict[str, Any]) -> str:
        raw_text = record.get("raw_document_text")
        if not isinstance(raw_text, str):
            raise ValueError(
                f"Raw baseline record {record.get('id')} is missing raw_document_text. "
                "Rerun the prepare stage to rebuild raw artifacts with source raw text."
            )
        if not raw_text:
            raise ValueError(
                f"Raw baseline record {record.get('id')} has empty raw_document_text. "
                "The raw artifact is invalid."
            )
        return raw_text

    def build_example(self, record: Dict[str, Any], *, include_target: bool) -> Dict[str, Any]:
        record_id = str(record.get("id") or "")
        prefix_ids = self._encode(self._render_prefix(record), context=f"raw prefix for {record_id}")
        query_ids = self._encode(self._render_query(record), context=f"raw query for {record_id}")
        target_ids = []
        if include_target:
            target_ids = self._encode(
                "<target>" + str(record.get("target_answer", {}).get("normalized_answer") or "") + "</target>",
                context=f"raw target for {record_id}",
            )
        input_ids: List[int] = []
        region_ids: List[int] = []

        def extend(ids: List[int], region_id: int) -> None:
            if not ids:
                return
            input_ids.extend(ids)
            region_ids.extend([region_id] * len(ids))

        extend(prefix_ids, -1)
        extend(query_ids, QUERY_REGION_ID)

        included_evidence_node_ids = self._paragraph_node_ids(record)
        raw_document_ids = self._encode(
            self._render_segment({"text": self._raw_document_text(record)}),
            context=f"raw document for {record_id}",
        )
        extend(raw_document_ids, -2)
        context_tokens = len(raw_document_ids)

        total_len = len(input_ids) + len(target_ids)
        if self.max_len > 0 and total_len > self.max_len:
            raise ValueError(
                f"Raw example {record_id} requires {total_len} tokens, exceeding max_len={self.max_len}. "
                "No truncation is allowed."
            )

        loss_suffix_labels = None
        loss_suffix_token_count = 0
        if include_target:
            extend(target_ids, TARGET_REGION_ID)
            labels = [
                token_id if region_id == TARGET_REGION_ID else -100
                for token_id, region_id in zip(input_ids, region_ids)
            ]
            # After the internal causal shift, the logits at the final
            # pre-target position and the target prefix positions predict the
            # target suffix tokens. Keeping only this window avoids allocating
            # full-vocab logits for the entire packed document.
            loss_suffix_labels = [-100] + list(target_ids)
            loss_suffix_token_count = len(loss_suffix_labels)
        else:
            labels = None

        efficiency_stats = {
            "sequence_len": float(len(input_ids)),
            "prefix_tokens": float(len(prefix_ids)),
            "query_tokens": float(len(query_ids)),
            "context_tokens": float(context_tokens),
            "target_tokens": float(len(target_ids)),
            "read_paragraphs": float(len(included_evidence_node_ids)),
        }
        return {
            "input_ids": input_ids,
            "region_ids": region_ids,
            "labels": labels,
            "loss_suffix_labels": loss_suffix_labels,
            "loss_suffix_token_count": int(loss_suffix_token_count),
            "record_id": record["id"],
            "included_evidence_node_ids": included_evidence_node_ids,
            "efficiency_stats": efficiency_stats,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build_example(self.records[idx], include_target=True)

    def generation_item(self, idx: int) -> Dict[str, Any]:
        return self.build_example(self.records[idx], include_target=False)


@dataclass
class RawCollator:
    tokenizer: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example["input_ids"]) for example in batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = []
        labels = []
        attention_mask = []
        for example in batch:
            seq_len = len(example["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(example["input_ids"] + [pad_id] * pad_len)
            if example.get("labels") is not None:
                labels.append(example["labels"] + [-100] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if labels:
            # For the default batch_size=1 setting, only the target suffix is
            # supervised. Qwen2 supports logits_to_keep, so we can avoid
            # materializing full-vocab logits for the entire 32k-token prefix.
            if (
                len(batch) == 1
                and batch[0].get("loss_suffix_labels") is not None
                and int(batch[0].get("loss_suffix_token_count") or 0) > 0
            ):
                output["labels"] = torch.tensor([batch[0]["loss_suffix_labels"]], dtype=torch.long)
                output["logits_to_keep"] = int(batch[0]["loss_suffix_token_count"])
            else:
                output["labels"] = torch.tensor(labels, dtype=torch.long)
        return output


def summarize_dataset_efficiency(dataset: RawQasperDataset, *, generation: bool) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for idx in range(len(dataset)):
        example = dataset.generation_item(idx) if generation else dataset[idx]
        count += 1
        for key, value in (example.get("efficiency_stats") or {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if count <= 0:
        return {}
    return {f"eff_{key}_avg": value / count for key, value in totals.items()}


def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    dataset: RawQasperDataset,
    *,
    device: str,
    out_dir: str,
    max_new_tokens: int,
) -> Dict[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    predictions_path = Path(out_dir) / "predictions.jsonl"
    metric_rows: List[Dict[str, float]] = []
    seen_questions: set[tuple[str, str]] = set()
    model.eval()
    target_end_token_id = strict_special_token_id(tokenizer, "</target>")
    eos_token_ids: List[int] = [target_end_token_id]
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) != target_end_token_id:
        eos_token_ids.insert(0, int(tokenizer.eos_token_id))
    with predictions_path.open("w", encoding="utf-8") as f_out:
        progress = tqdm(
            enumerate(dataset.records),
            total=len(dataset.records),
            desc="lm eval",
            unit="record",
        )
        for idx, record in progress:
            question_key = qasper_question_key(record)
            if question_key in seen_questions:
                continue
            seen_questions.add(question_key)
            example = dataset.generation_item(idx)
            input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0],
                )
            generated = outputs[0, input_ids.shape[1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            answer_scores = best_answer_scores(
                prediction,
                [str(item.get("normalized_answer") or "") for item in record.get("answer_refs", [])],
            )
            evidence_scores = best_evidence_scores(
                example.get("included_evidence_node_ids") or [],
                record.get("gold_evidence_sets") or [],
            )
            row = {
                "id": record["id"],
                "paper_id": question_key[0],
                "question_id": question_key[1],
                "prediction": prediction,
                "predicted_evidence_node_ids": example.get("included_evidence_node_ids") or [],
                "reference_answers": [str(item.get("normalized_answer") or "") for item in record.get("answer_refs", [])],
                **answer_scores,
                **evidence_scores,
            }
            metric_rows.append({key: float(value) for key, value in row.items() if key.endswith("_f1") or key.endswith("_em")})
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.set_postfix(wrote=len(metric_rows))

    metrics = aggregate_prediction_metrics(metric_rows)
    metrics.update(summarize_dataset_efficiency(dataset, generation=True))
    save_json(str(Path(out_dir) / "metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()
    model_path = resolve_local_hf_model_path(args.model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model_context_len = effective_model_max_length(tokenizer, model_config=model.config)
    if args.max_len > 0 and model_context_len > 0 and args.max_len > model_context_len:
        raise ValueError(
            f"--max-len={args.max_len} exceeds model context window {model_context_len}. "
            "No truncation is allowed."
        )
    effective_max_len = int(args.max_len) if int(args.max_len) > 0 else int(model_context_len)
    if effective_max_len <= 0:
        raise ValueError("Could not determine a valid max_len. Pass --max-len explicitly.")
    if int(args.batch) != 1:
        print(
            "[raw-baseline] suffix-logit memory optimization is only active for batch_size=1; "
            f"current batch={args.batch} will use the higher-memory full-logit path.",
            flush=True,
        )

    train_dataset = RawQasperDataset(
        args.data,
        tokenizer,
        max_len=effective_max_len,
        max_records=args.max_train,
        filter_include_target=True,
    )
    train_report = train_dataset.drop_report()
    print(
        "[raw-baseline] train "
        f"loaded={train_report['loaded_records']} kept={train_report['kept_records']} "
        f"dropped_over_limit={train_report['dropped_over_limit']} max_len={train_report['max_len']}",
        flush=True,
    )
    if train_report["dropped_examples"]:
        sample_ids = ", ".join(item["id"] for item in train_report["dropped_examples"])
        print(f"[raw-baseline] train dropped sample ids: {sample_ids}", flush=True)
    if len(train_dataset) <= 0:
        raise ValueError(
            f"No train records remain after dropping over-limit examples from {args.data}. "
            f"max_len={effective_max_len}"
        )
    eval_dataset = None
    if args.eval_data:
        eval_dataset = RawQasperDataset(
            args.eval_data,
            tokenizer,
            max_len=effective_max_len,
            max_records=args.max_eval,
            filter_include_target=False,
        )
        eval_report = eval_dataset.drop_report()
        print(
            "[raw-baseline] eval "
            f"loaded={eval_report['loaded_records']} kept={eval_report['kept_records']} "
            f"dropped_over_limit={eval_report['dropped_over_limit']} max_len={eval_report['max_len']}",
            flush=True,
        )
        if eval_report["dropped_examples"]:
            sample_ids = ", ".join(item["id"] for item in eval_report["dropped_examples"])
            print(f"[raw-baseline] eval dropped sample ids: {sample_ids}", flush=True)
        if len(eval_dataset) <= 0:
            print(
                f"[raw-baseline] eval dataset is empty after dropping over-limit examples from {args.eval_data}; "
                "skipping eval generation.",
                flush=True,
            )
            eval_dataset = None

    training_args = TrainingArguments(
        output_dir=str(Path(args.out) / "trainer_tmp"),
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RawCollator(tokenizer),
    )
    trainer.train()
    model.config.use_cache = True

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    if eval_dataset is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        metrics = evaluate_generation(
            model,
            tokenizer,
            eval_dataset,
            device=device,
            out_dir=args.out,
            max_new_tokens=args.gen_max_new_tokens,
        )
        print(metrics)


if __name__ == "__main__":
    main()
