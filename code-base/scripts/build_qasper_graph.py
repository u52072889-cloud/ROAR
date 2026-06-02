#!/usr/bin/env python
"""Build QASPER graph and raw-baseline artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from data_utils import GRAPH_SCHEMA_VERSION
from graph_utils import build_structural_edges, dedupe_edges
from qasper_utils import (
    build_paper_nodes,
    load_qasper_rows,
    question_records_for_paper,
    render_linearized_document,
    render_raw_document_text,
)


def clone_nodes(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for node in nodes:
        copied = dict(node)
        copied["position_features"] = dict(node.get("position_features") or {})
        cloned.append(copied)
    return cloned


def clone_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(edge) for edge in edges]


def enrich_record(
    record: Dict[str, Any],
    *,
    nodes: Sequence[Dict[str, Any]],
    routing_links: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(record)
    out["nodes"] = clone_nodes(nodes)
    out["routing_links"] = clone_edges(routing_links)
    out["pointer_links"] = []
    out["query"] = dict(record.get("query") or {})
    out["query"]["seed_candidates"] = []
    gold_sets = [
        [int(node_id) for node_id in item.get("evidence_node_ids") or []]
        for item in out.get("answer_refs", [])
        if item.get("has_text_evidence")
    ]
    out["gold_evidence_sets"] = gold_sets
    section_count = len(
        {
            int(node["section_index"])
            for node in out["nodes"]
            if int(node.get("section_index", -1)) >= 0
        }
    )
    paragraph_count = sum(1 for node in out["nodes"] if node.get("node_type") == "paragraph")
    out["meta"] = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "num_nodes": len(out["nodes"]),
        "num_sections": section_count,
        "num_paragraphs": paragraph_count,
        "num_routing_links": len(out["routing_links"]),
        "num_seed_candidates": len(out["query"].get("seed_candidates") or []),
        "num_gold_evidence": len(out.get("gold_evidence_node_ids") or []),
        "has_text_evidence": bool(out.get("gold_evidence_node_ids")),
        "has_text_evidence_refs": any(item.get("has_text_evidence") for item in out.get("answer_refs", [])),
    }
    out["graph_schema_version"] = GRAPH_SCHEMA_VERSION
    return out


def raw_artifact(record: Dict[str, Any], *, paper: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    out["linearized_context"] = render_linearized_document(record)
    out["raw_document_text"] = render_raw_document_text(paper)
    if not out["raw_document_text"]:
        raise ValueError(
            f"Paper {record.get('paper_id')} produced empty raw_document_text. "
            "The raw baseline requires source full_text content."
        )
    raw_segments: List[Dict[str, Any]] = []
    last_section_index: int | None = None
    for node in record.get("nodes", []):
        if node.get("node_type") != "paragraph":
            continue
        section_index = int(node.get("section_index", -1))
        if last_section_index != section_index:
            section_title = str(node.get("section_title") or "").strip()
            raw_segments.append(
                {
                    "node_id": int(node["node_id"]),
                    "segment_type": "section_header",
                    "text": f"Section: {section_title}" if section_title else f"Section {section_index + 1}",
                }
            )
            last_section_index = section_index
        raw_segments.append(
            {
                "node_id": int(node["node_id"]),
                "segment_type": "paragraph",
                "text": str(node.get("text") or ""),
            }
        )
    out["raw_segments"] = raw_segments
    return out


def build_split_records(
    rows: Sequence[Dict[str, Any]],
    *,
    split: str,
    limit: int,
    max_section_chars: int,
    keep_unanswerable: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    graph_records: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []
    processed_papers = 0

    iterator = tqdm(rows, desc=f"build {split}", unit="papers")
    for paper in iterator:
        processed_papers += 1
        nodes = build_paper_nodes(paper, max_section_chars=max_section_chars)
        if not nodes:
            if limit and processed_papers >= limit:
                break
            continue
        routing_links = dedupe_edges(list(build_structural_edges(nodes)))
        for record in question_records_for_paper(
            paper,
            split=split,
            nodes=nodes,
            keep_unanswerable=keep_unanswerable,
        ):
            enriched = enrich_record(
                record,
                nodes=nodes,
                routing_links=routing_links,
            )
            graph_records.append(enriched)
            raw_records.append(raw_artifact(enriched, paper=paper))
            iterator.set_postfix(records=len(graph_records))
        if limit and processed_papers >= limit:
            break
    return graph_records, raw_records, processed_papers


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/qasper")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-limit", type=int, default=0, help="Max train papers to process. 0 means all papers.")
    parser.add_argument("--eval-limit", type=int, default=0, help="Max eval papers to process. 0 means all papers.")
    parser.add_argument("--max-section-chars", type=int, default=0)
    parser.add_argument("--drop-unanswerable", action="store_true")
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-eval", required=True)
    parser.add_argument("--out-train-raw", required=True)
    parser.add_argument("--out-eval-raw", required=True)
    args = parser.parse_args()

    if int(args.max_section_chars) > 0:
        raise ValueError(
            f"--max-section-chars={args.max_section_chars} would truncate section text. "
            "Set it to 0 to disable truncation."
        )
    train_rows = load_qasper_rows(
        args.dataset,
        split=args.train_split,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )
    eval_rows = load_qasper_rows(
        args.dataset,
        split=args.eval_split,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )
    train_graph, train_raw, train_papers = build_split_records(
        train_rows,
        split=args.train_split,
        limit=args.train_limit,
        max_section_chars=args.max_section_chars,
        keep_unanswerable=not args.drop_unanswerable,
    )
    eval_graph, eval_raw, eval_papers = build_split_records(
        eval_rows,
        split=args.eval_split,
        limit=args.eval_limit,
        max_section_chars=args.max_section_chars,
        keep_unanswerable=not args.drop_unanswerable,
    )
    write_jsonl(args.out_train, train_graph)
    write_jsonl(args.out_eval, eval_graph)
    write_jsonl(args.out_train_raw, train_raw)
    write_jsonl(args.out_eval_raw, eval_raw)

    summary = {
        "train_papers": train_papers,
        "train_records": len(train_graph),
        "eval_papers": eval_papers,
        "eval_records": len(eval_graph),
        "out_train": args.out_train,
        "out_eval": args.out_eval,
        "out_train_raw": args.out_train_raw,
        "out_eval_raw": args.out_eval_raw,
    }
    print(summary)


if __name__ == "__main__":
    main()
