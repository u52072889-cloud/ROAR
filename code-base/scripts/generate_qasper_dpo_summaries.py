#!/usr/bin/env python
"""Generate node summaries on QASPER graph artifacts with a DPO-tuned model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from build_qasper_dpo_data import (  # noqa: E402
    GENERATOR_SYSTEM_PROMPT,
    build_node_generation_prompt,
    decode_generations,
    format_chat_prompt,
    model_load_source,
    require_nonempty_tokenizer,
    require_texts_within_limit,
)
from data_utils import (  # noqa: E402
    GRAPH_SCHEMA_VERSION,
    compact_whitespace,
    effective_model_max_length,
    normalize_node_tokens,
    resolve_local_hf_model_path,
    set_offline_hf_env,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT = REPO_ROOT / "artifacts" / "qasper_test_graph.jsonl"
MAX_NODE_PHRASES = 5
SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")
OUTPUT_SPLIT_RE = re.compile(r"\n+")


def record_paper_id(record: Dict[str, Any]) -> str:
    paper_id = compact_whitespace(record.get("paper_id") or "")
    if not paper_id:
        raise ValueError(f"Record {record.get('id')} is missing paper_id.")
    return paper_id


def load_generation_model(
    *,
    model_name_or_path: str,
    adapter_path: str,
    device: str,
    cache_dir: str,
    allow_download: bool,
) -> tuple[Any, Any]:
    tokenizer_source = adapter_path if str(adapter_path or "").strip() else model_name_or_path
    tokenizer_path = resolve_local_hf_model_path(tokenizer_source, cache_dir=cache_dir)
    tokenizer_load_source = model_load_source(
        tokenizer_source,
        str(tokenizer_path),
        allow_download=allow_download,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_load_source,
        use_fast=True,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token.")
    require_nonempty_tokenizer(tokenizer, context=f"Tokenizer source '{tokenizer_source}'")

    base_model_path = resolve_local_hf_model_path(model_name_or_path, cache_dir=cache_dir)
    base_model_is_local = Path(base_model_path).exists()
    if not base_model_is_local and not allow_download:
        raise FileNotFoundError(
            f"Model '{model_name_or_path}' was not found in cache_dir='{cache_dir}' while offline mode is enabled. "
            "Pre-download the model into the cache or rerun with --allow-download."
        )
    base_model_source = model_load_source(
        model_name_or_path,
        str(base_model_path),
        allow_download=allow_download,
    )
    model_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        local_files_only=not allow_download,
        **model_kwargs,
    )
    if str(adapter_path or "").strip():
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_texts(
    *,
    tokenizer: Any,
    model: Any,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
) -> List[str]:
    if not prompts:
        return []
    context_window = effective_model_max_length(tokenizer, model_config=model.config)
    require_texts_within_limit(
        tokenizer=tokenizer,
        texts=prompts,
        limit=context_window,
        context="DPO summary generation prompt",
        reserve_tokens=max_new_tokens,
    )
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_generations(
        tokenizer=tokenizer,
        outputs=outputs,
        attention_mask=inputs["attention_mask"],
    )


def split_raw_output(raw_output: str) -> tuple[str, List[str]]:
    raw_text = str(raw_output or "").strip()
    if not raw_text:
        raise ValueError("Generated empty node summary.")
    phrases: List[str] = []
    for part in OUTPUT_SPLIT_RE.split(raw_text):
        phrase = compact_whitespace(part)
        if not phrase:
            continue
        phrases.append(phrase)
        if len(phrases) >= MAX_NODE_PHRASES:
            break
    if not phrases:
        phrases = [compact_whitespace(raw_text)]
    return raw_text, phrases


def joined_tokens(tokens: List[str]) -> str:
    return " ".join(str(token) for token in tokens if str(token).strip())


def build_summary_bundle(
    *,
    record: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    all_nodes = list(record.get("nodes") or [])
    if not all_nodes:
        raise ValueError(f"Record {record.get('id')} has no nodes.")
    nodes = [
        dict(node)
        for node in all_nodes
        if str(node.get("node_type") or "") == "paragraph"
    ]
    if not nodes:
        raise ValueError(f"Record {record.get('id')} has no paragraph nodes.")
    prompts: List[str] = []
    prompt_items: List[Dict[str, int]] = []
    node_raw_outputs: List[str] = ["" for _ in nodes]
    node_phrases: List[List[str]] = [[] for _ in nodes]
    for node_index, node in enumerate(nodes):
        node_text = compact_whitespace(str(node.get("text") or ""))
        if not node_text:
            continue
        prompts.append(
            format_chat_prompt(
                tokenizer,
                GENERATOR_SYSTEM_PROMPT,
                build_node_generation_prompt(
                    section_title=str(node.get("section_title") or ""),
                    node_text=node_text,
                ),
            )
        )
        prompt_items.append({"node_index": int(node_index)})
    outputs: List[str] = []
    step = max(1, int(batch_size))
    for start in range(0, len(prompts), step):
        batch_prompts = prompts[start : start + step]
        outputs.extend(
            generate_texts(
                tokenizer=tokenizer,
                model=model,
                prompts=batch_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        )
    if len(outputs) != len(prompt_items):
        raise ValueError(f"Expected {len(prompt_items)} node outputs, received {len(outputs)}.")
    for item, raw_output in zip(prompt_items, outputs):
        node_index = int(item["node_index"])
        raw_text, phrases = split_raw_output(raw_output)
        node_raw_outputs[node_index] = raw_text
        node_phrases[node_index] = list(phrases[:MAX_NODE_PHRASES])
    node_summaries: List[Dict[str, Any]] = []
    for node_index, node in enumerate(nodes):
        raw_text = node_raw_outputs[node_index]
        phrases = node_phrases[node_index]
        node_tokens = normalize_node_tokens(phrases, max_tokens=MAX_NODE_PHRASES) if phrases else []
        node_summaries.append(
            {
                "node_id": int(node["node_id"]),
                "node_type": str(node.get("node_type") or ""),
                "section_title": str(node.get("section_title") or ""),
                "node_sum_raw_output": raw_text,
                "node_sum_text": "; ".join(phrases),
                "node_sum_tokens": list(node_tokens),
                "node_sum_tokens_text": joined_tokens(node_tokens),
                "pointer_candidates": [],
                "global_pointer_candidates": [],
                "pointer_links": [],
                "num_pointer_candidates": 0,
                "num_pointer_links": 0,
            }
        )
    return {
        "node_summaries": node_summaries,
        "pointer_candidates": [],
        "global_pointer_candidates": [],
        "pointer_links": [],
    }


def enrich_record(
    *,
    record: Dict[str, Any],
    bundle: Dict[str, Any],
    summary_source: str,
) -> Dict[str, Any]:
    out = dict(record)
    out["graph_schema_version"] = GRAPH_SCHEMA_VERSION
    out["query"] = dict(record.get("query") or {})
    out["query"]["seed_candidates"] = []
    out["query"]["node_scores"] = []
    out["node_summaries"] = copy.deepcopy(bundle["node_summaries"])
    out["pointer_candidates"] = copy.deepcopy(bundle["pointer_candidates"])
    out["global_pointer_candidates"] = copy.deepcopy(bundle["global_pointer_candidates"])
    out["pointer_links"] = copy.deepcopy(bundle["pointer_links"])
    out["summary_source"] = summary_source
    out["meta"] = dict(record.get("meta") or {})
    out["meta"]["graph_schema_version"] = GRAPH_SCHEMA_VERSION
    out["meta"]["num_pointer_candidates"] = 0
    out["meta"]["num_global_pointer_candidates"] = 0
    out["meta"]["num_pointer_links"] = 0
    out["meta"]["num_seed_candidates"] = 0
    return out


def sanitize_name(value: str) -> str:
    return SANITIZE_RE.sub("_", str(value or "")).strip("_").lower()


def build_summary_source(*, model_name_or_path: str, adapter_path: str) -> str:
    model_part = sanitize_name(model_name_or_path)
    adapter_part = sanitize_name(Path(adapter_path).name if str(adapter_path or "").strip() else "base")
    return f"hf_{model_part}_dpo_{adapter_part}_qasper_sum_score_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--paper-summary-cache-size", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if int(args.paper_summary_cache_size) < 0:
        raise ValueError("--paper-summary-cache-size must be non-negative.")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative.")

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_generation_model(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        device=device,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )
    summary_source = build_summary_source(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    processed = 0
    skipped_existing = 0
    cache_hits = 0
    cache_misses = 0
    paper_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    existing_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f_existing:
            for line in f_existing:
                row = json.loads(line)
                existing_ids.add(str(row.get("id") or ""))
    out_mode = "a" if args.resume else "w"
    with open(args.data, "r", encoding="utf-8") as f_in, out_path.open(out_mode, encoding="utf-8", buffering=1) as f_out:
        iterator = tqdm(f_in, desc=f"dpo summaries {Path(args.data).name}", unit="records")
        for line in iterator:
            record = json.loads(line)
            processed += 1
            if int(args.max_records) > 0 and processed > int(args.max_records):
                break
            record_id = str(record.get("id") or "")
            if args.resume and record_id in existing_ids:
                skipped_existing += 1
                iterator.set_postfix(
                    wrote=wrote,
                    skipped_existing=skipped_existing,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                )
                continue
            bundle = None
            if int(args.paper_summary_cache_size) > 0:
                cache_key = record_paper_id(record)
                bundle = paper_cache.get(cache_key)
                if bundle is not None:
                    paper_cache.move_to_end(cache_key)
                    cache_hits += 1
                else:
                    cache_misses += 1
            if bundle is None:
                bundle = build_summary_bundle(
                    record=record,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=int(args.batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                )
                if int(args.paper_summary_cache_size) > 0:
                    cache_key = record_paper_id(record)
                    paper_cache[cache_key] = bundle
                    paper_cache.move_to_end(cache_key)
                    while len(paper_cache) > int(args.paper_summary_cache_size):
                        paper_cache.popitem(last=False)
            enriched = enrich_record(record=record, bundle=bundle, summary_source=summary_source)
            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            wrote += 1
            if args.resume:
                existing_ids.add(record_id)
            iterator.set_postfix(
                wrote=wrote,
                skipped_existing=skipped_existing,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
    print(
        json.dumps(
            {
                "wrote": wrote,
                "out": str(out_path),
                "summary_source": summary_source,
                "resume": bool(args.resume),
                "skipped_existing": skipped_existing,
                "paper_summary_cache_hits": cache_hits,
                "paper_summary_cache_misses": cache_misses,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
