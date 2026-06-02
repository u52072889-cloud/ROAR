#!/usr/bin/env python
"""Recompute QASPER seed candidates with embedding similarity over DPO node summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from data_utils import compact_whitespace, resolve_local_hf_model_path, set_offline_hf_env  # noqa: E402


DEFAULT_SEED_CANDIDATE_K = 12


def question_text(record: Dict[str, Any]) -> str:
    query = record.get("query") or {}
    if isinstance(query, dict):
        return compact_whitespace(query.get("question") or "")
    return compact_whitespace(query)


def summary_text(summary: Dict[str, Any]) -> str:
    node_sum_text = compact_whitespace(summary.get("node_sum_text") or "")
    if not node_sum_text:
        raise ValueError(f"Node summary {summary.get('node_id')} is missing node_sum_text.")
    section_title = compact_whitespace(summary.get("section_title") or "")
    return compact_whitespace("\n".join(part for part in [section_title, node_sum_text] if part))


def load_records(path: Path, *, max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if max_records and len(records) >= max_records:
                break
            record = json.loads(line)
            question = question_text(record)
            if not question:
                raise ValueError(f"{path}:{line_number} record {record.get('id')} is missing query.question.")
            node_items: List[Dict[str, Any]] = []
            seen_node_ids: set[int] = set()
            for summary in record.get("node_summaries") or []:
                if str(summary.get("node_type") or "") != "paragraph":
                    continue
                node_id = int(summary["node_id"])
                if node_id in seen_node_ids:
                    raise ValueError(f"{path}:{line_number} record {record.get('id')} has duplicate summary for node {node_id}.")
                text = summary_text(summary)
                if not text:
                    raise ValueError(f"{path}:{line_number} record {record.get('id')} node {node_id} has empty summary text.")
                seen_node_ids.add(node_id)
                node_items.append({"node_id": node_id, "text": text})
            if not node_items:
                raise ValueError(f"{path}:{line_number} record {record.get('id')} has no paragraph node summaries.")
            records.append({"record": record, "question": question, "nodes": node_items})
    return records


def build_text_index(records: Sequence[Dict[str, Any]]) -> tuple[List[str], Dict[str, int]]:
    texts: List[str] = []
    text_to_index: Dict[str, int] = {}

    def add(text: str) -> None:
        if text not in text_to_index:
            text_to_index[text] = len(texts)
            texts.append(text)

    for item in records:
        add(str(item["question"]))
        for node in item["nodes"]:
            add(str(node["text"]))
    return texts, text_to_index


def load_embedder(*, model_name: str, cache_dir: str, device: str, allow_download: bool) -> tuple[Any, Any]:
    model_path = resolve_local_hf_model_path(model_name, cache_dir=cache_dir)
    model_is_local = Path(model_path).exists()
    if not model_is_local and not allow_download:
        raise FileNotFoundError(
            f"Embedding model '{model_name}' was not found in cache_dir='{cache_dir}'. "
            "Pre-download it or rerun with --allow-download."
        )
    source = model_name if allow_download and not model_is_local else str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def embed_texts(
    *,
    tokenizer: Any,
    model: Any,
    texts: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embed seed texts", unit="batch"):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        vectors.append(F.normalize(summed / denom, dim=-1).cpu())
    return torch.cat(vectors, dim=0)


def enrich_records(
    *,
    records: Sequence[Dict[str, Any]],
    embeddings: torch.Tensor,
    text_to_index: Dict[str, int],
    seed_candidate_k: int,
    embedder: str,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in tqdm(records, desc="seed candidates", unit="records"):
        record = dict(item["record"])
        query_vector = embeddings[text_to_index[str(item["question"])]]
        node_indices = [text_to_index[str(node["text"])] for node in item["nodes"]]
        node_vectors = embeddings[node_indices]
        scores = torch.mv(node_vectors, query_vector).tolist()
        scored_nodes = [
            {
                "node_id": int(node["node_id"]),
                "score": float(score),
            }
            for node, score in zip(item["nodes"], scores)
        ]
        scored_nodes.sort(key=lambda row: (-float(row["score"]), int(row["node_id"])))
        for rank, row in enumerate(scored_nodes, start=1):
            row["rank"] = int(rank)
        seed_candidates = [dict(row) for row in scored_nodes[: int(seed_candidate_k)]]
        query = dict(record.get("query") or {})
        query["seed_candidates"] = seed_candidates
        query["node_scores"] = scored_nodes
        record["query"] = query
        meta = dict(record.get("meta") or {})
        meta["num_seed_candidates"] = len(seed_candidates)
        record["meta"] = meta
        enriched.append(record)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed-candidate-k", type=int, default=DEFAULT_SEED_CANDIDATE_K)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_length) <= 0:
        raise ValueError("--max-length must be positive.")
    if int(args.seed_candidate_k) <= 0:
        raise ValueError("--seed-candidate-k must be positive.")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative.")

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(Path(args.data), max_records=int(args.max_records))
    if not records:
        raise ValueError(f"No records loaded from {args.data}.")
    tokenizer, model = load_embedder(
        model_name=str(args.embedder),
        cache_dir=str(args.cache_dir),
        device=device,
        allow_download=bool(args.allow_download),
    )
    texts, text_to_index = build_text_index(records)
    embeddings = embed_texts(
        tokenizer=tokenizer,
        model=model,
        texts=texts,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    enriched = enrich_records(
        records=records,
        embeddings=embeddings,
        text_to_index=text_to_index,
        seed_candidate_k=int(args.seed_candidate_k),
        embedder=str(args.embedder),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in enriched:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "records": len(enriched),
                "out": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
