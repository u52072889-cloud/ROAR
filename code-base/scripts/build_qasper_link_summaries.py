#!/usr/bin/env python
"""Build natural-language routing link summaries from existing QASPER summary artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from build_qasper_dpo_data import format_chat_prompt  # noqa: E402
from data_utils import compact_whitespace, set_offline_hf_env  # noqa: E402
from generate_qasper_dpo_summaries import generate_texts, load_generation_model  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[1] / ".hf" / "hub")
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_PAPER_CACHE_SIZE = 0
DEFAULT_MAX_RECORDS = 0
DEFAULT_GENERATOR_DEVICE = "cuda"

LINK_LABELS = [
    "DATASET_INFO",
    "BASELINE_COMPARISON",
    "EVALUATION",
    "ANNOTATION_SCHEMA",
    "METHOD_DETAILS",
    "RESULTS",
    "EXAMPLE_CASE",
    "OTHER",
]

LINK_LABEL_TO_PHRASE = {
    "DATASET_INFO": "gives dataset information",
    "BASELINE_COMPARISON": "names baselines or comparisons",
    "EVALUATION": "gives evaluation setup or metrics",
    "ANNOTATION_SCHEMA": "gives labels or annotation scheme",
    "METHOD_DETAILS": "explains method or features",
    "RESULTS": "reports results or findings",
    "EXAMPLE_CASE": "gives example or case",
    "OTHER": "continues related content",
}

LINK_SUMMARY_SYSTEM_PROMPT = (
    "Classify the destination paragraph for graph routing. "
    "Output exactly one label from this list: "
    + ", ".join(LINK_LABELS)
    + ". Do not output any other words."
)

LEADING_MARKUP_RE = re.compile(r"^\s*(?:[-*]|[0-9]+[.)])\s*")
MAX_LABEL_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="Input summary JSONL. Repeat to process multiple files.")
    parser.add_argument("--output", action="append", default=[], help="Output JSONL. Repeat to match each --input.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=DEFAULT_GENERATOR_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--paper-cache-size", type=int, default=DEFAULT_PAPER_CACHE_SIZE)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    return parser.parse_args()


def resolve_jobs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    inputs = [Path(value) for value in args.input]
    outputs = [Path(value) for value in args.output]
    if not inputs or not outputs or len(inputs) != len(outputs):
        raise ValueError("Provide matching repeated --input and --output arguments.")
    return list(zip(inputs, outputs))


def count_input_records(path: Path, *, max_records: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            count += 1
            if int(max_records) and count >= int(max_records):
                break
    return count


def node_text_index(record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for node in record.get("nodes") or []:
        node_id = int(node["node_id"])
        index[node_id] = dict(node)
    if not index:
        raise ValueError(f"Record {record.get('id')} is missing nodes.")
    return index


def edge_key(*, src: int, dst: int, edge_type: str) -> Tuple[int, int, str]:
    return (int(src), int(dst), str(edge_type or ""))


def build_link_generation_prompt(
    *,
    src_section_title: str,
    src_text: str,
    dst_section_title: str,
    dst_text: str,
) -> str:
    src_title = compact_whitespace(src_section_title) or "Unknown"
    dst_title = compact_whitespace(dst_section_title) or "Unknown"
    src_body = compact_whitespace(src_text)
    dst_body = compact_whitespace(dst_text)
    if not src_body or not dst_body:
        raise ValueError("Cannot build a link generation prompt for empty paragraph text.")
    return "\n".join(
        [
            "Choose the best label for the destination paragraph.",
            "Labels: " + ", ".join(LINK_LABELS),
            "",
            "Source paragraph:",
            src_title,
            src_body,
            "",
            "Destination paragraph:",
            dst_title,
            dst_body,
        ]
    )


def build_retry_generation_prompt(*, base_prompt: str, invalid_output: str) -> str:
    invalid_text = compact_whitespace(str(invalid_output or "")) or "<empty>"
    return "\n".join(
        [
            "The previous output was invalid because it was not exactly one allowed label.",
            "Previous output:",
            invalid_text,
            "",
            "Try again. Output exactly one label from the list and no other words.",
            "",
            base_prompt,
        ]
    )


def parse_link_label(raw_output: str) -> str | None:
    candidate = LEADING_MARKUP_RE.sub("", str(raw_output or "").strip())
    candidate = compact_whitespace(candidate).upper().replace("-", "_")
    candidate = candidate.strip().strip(".,:;`'\"[](){}")
    if candidate in LINK_LABEL_TO_PHRASE:
        return candidate
    return None


def clean_raw_link_output(raw_output: str) -> str:
    return compact_whitespace(LEADING_MARKUP_RE.sub("", str(raw_output or "").strip()))


def prefix_direction(*, edge_type: str, label: str | None, raw_output: str) -> str:
    direction = "next paragraph" if str(edge_type or "") == "paragraph_next" else "previous paragraph"
    if label in LINK_LABEL_TO_PHRASE:
        phrase = LINK_LABEL_TO_PHRASE[str(label)]
    else:
        phrase = clean_raw_link_output(raw_output)
    if not phrase:
        return direction
    return compact_whitespace(f"{direction} {phrase}")


def generate_link_outputs_with_retries(
    *,
    tokenizer: Any,
    model: Any,
    items: List[Dict[str, Any]],
    prompts: List[str],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [
        {"label": None, "raw_output": "", "raw_outputs": [], "attempts": 0} for _ in prompts
    ]
    pending = list(range(len(prompts)))
    active_prompts = list(prompts)
    step = max(1, int(batch_size))

    for attempt in range(1, MAX_LABEL_ATTEMPTS + 1):
        if not pending:
            break
        attempt_prompts = [active_prompts[index] for index in pending]
        outputs: List[str] = []
        for start in range(0, len(attempt_prompts), step):
            outputs.extend(
                generate_texts(
                    tokenizer=tokenizer,
                    model=model,
                    prompts=attempt_prompts[start : start + step],
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
            )
        if len(outputs) != len(pending):
            raise ValueError(f"Expected {len(pending)} link summary outputs, received {len(outputs)}.")

        next_pending: List[int] = []
        for index, raw_output in zip(pending, outputs):
            raw_text = compact_whitespace(str(raw_output or ""))
            label = parse_link_label(raw_text)
            results[index]["raw_output"] = raw_text
            results[index]["raw_outputs"].append(raw_text)
            results[index]["label"] = label
            results[index]["attempts"] = attempt
            if label is None and attempt < MAX_LABEL_ATTEMPTS:
                retry_prompt = build_retry_generation_prompt(
                    base_prompt=str(items[index]["base_prompt"]),
                    invalid_output=raw_text,
                )
                active_prompts[index] = format_chat_prompt(tokenizer, LINK_SUMMARY_SYSTEM_PROMPT, retry_prompt)
                next_pending.append(index)
        pending = next_pending
    return results


def build_paper_link_bundle(
    *,
    record: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    nodes = node_text_index(record)
    prompts: List[str] = []
    items: List[Dict[str, Any]] = []
    seen = set()
    for summary in record.get("node_summaries") or []:
        src_id = int(summary["node_id"])
        src_node = nodes.get(src_id)
        if src_node is None:
            raise ValueError(f"Record {record.get('id')} is missing node payload for src node {src_id}.")
        links = summary.get("pointer_links")
        if not isinstance(links, list):
            raise ValueError(f"Record {record.get('id')} node {src_id} is missing pointer_links.")
        for link in links:
            dst_id = int(link["dst"])
            edge_type = str(link.get("edge_type") or link.get("type") or "")
            key = edge_key(src=src_id, dst=dst_id, edge_type=edge_type)
            if key in seen:
                continue
            dst_node = nodes.get(dst_id)
            if dst_node is None:
                raise ValueError(f"Record {record.get('id')} is missing node payload for dst node {dst_id}.")
            seen.add(key)
            base_prompt = build_link_generation_prompt(
                src_section_title=str(src_node.get("section_title") or summary.get("section_title") or ""),
                src_text=str(src_node.get("text") or ""),
                dst_section_title=str(dst_node.get("section_title") or ""),
                dst_text=str(dst_node.get("text") or ""),
            )
            prompts.append(format_chat_prompt(tokenizer, LINK_SUMMARY_SYSTEM_PROMPT, base_prompt))
            items.append(
                {
                    "key": key,
                    "edge_type": edge_type,
                    "base_prompt": base_prompt,
                }
            )

    outputs = generate_link_outputs_with_retries(
        tokenizer=tokenizer,
        model=model,
        items=items,
        prompts=prompts,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if len(outputs) != len(items):
        raise ValueError(f"Expected {len(items)} link summary outputs, received {len(outputs)}.")

    bundle: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for item, result in zip(items, outputs):
        label = result.get("label")
        raw_output = compact_whitespace(str(result.get("raw_output") or ""))
        link_phrase = prefix_direction(edge_type=str(item["edge_type"]), label=label, raw_output=raw_output)
        bundle[item["key"]] = {
            "link_sum_raw_output": raw_output,
            "link_sum_raw_outputs": list(result.get("raw_outputs") or []),
            "link_label": str(label or ""),
            "link_generation_attempts": int(result.get("attempts") or 0),
            "link_tokens": [link_phrase],
            "link_tokens_text": link_phrase,
        }
    return bundle


def flatten_pointer_links(node_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for summary in node_summaries:
        src = int(summary["node_id"])
        for link in summary.get("pointer_links") or []:
            row = dict(link)
            row["src"] = int(row.get("src") or src)
            row["dst"] = int(row["dst"])
            flattened.append(row)
    return flattened


def enrich_record(
    *,
    record: Dict[str, Any],
    paper_bundle: Dict[Tuple[int, int, str], Dict[str, Any]],
    model_name_or_path: str,
) -> Dict[str, Any]:
    out = dict(record)
    summaries = [dict(item) for item in record.get("node_summaries") or []]
    if not summaries:
        raise ValueError(f"Record {record.get('id')} is missing node_summaries.")
    for summary in summaries:
        src_id = int(summary["node_id"])
        links = summary.get("pointer_links")
        if not isinstance(links, list):
            raise ValueError(f"Record {record.get('id')} node {src_id} is missing pointer_links.")
        rewritten_links: List[Dict[str, Any]] = []
        for link in links:
            dst_id = int(link["dst"])
            edge_type = str(link.get("edge_type") or link.get("type") or "")
            payload = paper_bundle.get(edge_key(src=src_id, dst=dst_id, edge_type=edge_type))
            if payload is None:
                raise ValueError(
                    f"Record {record.get('id')} is missing generated link summary for edge {src_id}->{dst_id} ({edge_type})."
                )
            enriched = dict(link)
            enriched["src"] = src_id
            enriched["dst"] = dst_id
            enriched["type"] = str(link.get("type") or edge_type)
            enriched["edge_type"] = edge_type
            enriched.update(payload)
            rewritten_links.append(enriched)
        summary["pointer_links"] = rewritten_links
        summary["num_pointer_links"] = len(rewritten_links)

    out["node_summaries"] = summaries
    flattened_links = flatten_pointer_links(summaries)
    out["pointer_links"] = flattened_links
    meta = dict(out.get("meta") or {})
    meta["num_pointer_links"] = len(flattened_links)
    meta["num_nonempty_link_summaries"] = sum(1 for link in flattened_links if link.get("link_tokens"))
    meta["link_summary_source"] = "qwen2.5_3b_8way_label"
    meta["link_summary_model"] = str(model_name_or_path)
    out["meta"] = meta
    return out


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    model_name_or_path: str,
    batch_size: int,
    max_new_tokens: int,
    paper_cache_size: int,
    max_records: int,
) -> Dict[str, float]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = count_input_records(input_path, max_records=max_records)
    record_count = 0
    link_count = 0
    nonempty_count = 0
    cache_hits = 0
    cache_misses = 0
    paper_cache: OrderedDict[str, Dict[Tuple[int, int, str], Dict[str, Any]]] = OrderedDict()

    progress = tqdm(
        total=total_records,
        desc=f"linksum {input_path.name}",
        unit="record",
        dynamic_ncols=True,
    )
    try:
        with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if int(max_records) and record_count >= int(max_records):
                    break
                record = json.loads(line)
                paper_id = compact_whitespace(record.get("paper_id") or "") or str(record.get("id") or line_number)
                paper_bundle = paper_cache.get(paper_id)
                if paper_bundle is None:
                    paper_bundle = build_paper_link_bundle(
                        record=record,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                    )
                    cache_misses += 1
                    paper_cache[paper_id] = paper_bundle
                    if int(paper_cache_size) > 0:
                        while len(paper_cache) > int(paper_cache_size):
                            paper_cache.popitem(last=False)
                else:
                    cache_hits += 1
                    paper_cache.move_to_end(paper_id)

                out = enrich_record(record=record, paper_bundle=paper_bundle, model_name_or_path=model_name_or_path)
                sink.write(json.dumps(out, ensure_ascii=False) + "\n")
                record_count += 1
                links = out.get("pointer_links") or []
                link_count += len(links)
                nonempty_count += sum(1 for link in links if link.get("link_tokens"))
                progress.update(1)
    finally:
        progress.close()

    return {
        "records": float(record_count),
        "links": float(link_count),
        "nonempty": float(nonempty_count),
        "paper_cache_hits": float(cache_hits),
        "paper_cache_misses": float(cache_misses),
    }


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if int(args.paper_cache_size) < 0:
        raise ValueError("--paper-cache-size must be non-negative.")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative.")

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()

    device = str(args.device or DEFAULT_GENERATOR_DEVICE)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    tokenizer, model = load_generation_model(
        model_name_or_path=str(args.model),
        adapter_path="",
        device=device,
        cache_dir=str(args.cache_dir),
        allow_download=bool(args.allow_download),
    )
    jobs = resolve_jobs(args)
    for input_path, output_path in tqdm(jobs, desc="linksum files", unit="file", dynamic_ncols=True):
        stats = process_file(
            input_path=input_path,
            output_path=output_path,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_name_or_path=str(args.model),
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            paper_cache_size=int(args.paper_cache_size),
            max_records=int(args.max_records),
        )
        print(
            f"{input_path} -> {output_path} | "
            f"records={int(stats['records'])} links={int(stats['links'])} "
            f"nonempty={int(stats['nonempty'])} "
            f"paper_cache_hits={int(stats['paper_cache_hits'])} "
            f"paper_cache_misses={int(stats['paper_cache_misses'])}"
        )


if __name__ == "__main__":
    main()
