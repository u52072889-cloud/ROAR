#!/usr/bin/env python
"""Build structural pointer links from precomputed node summaries."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from data_utils import GRAPH_SCHEMA_VERSION, compact_whitespace
from graph_utils import EDGE_PRIORITY, build_routing_link_index, node_lookup, relation_text_for_edge


STRUCTURAL_CANDIDATE_SOURCE = "structural"


def _edge_type(edge: Dict[str, Any]) -> str:
    return str(edge.get("edge_type") or edge.get("type") or "").strip()


def normalize_pointer_edge(
    edge: Dict[str, Any],
    *,
    src_node: Dict[str, Any],
    dst_node: Dict[str, Any],
) -> Dict[str, Any]:
    edge_type = _edge_type(edge)
    normalized = dict(edge)
    normalized["src"] = int(src_node["node_id"])
    normalized["dst"] = int(edge["dst"])
    normalized["type"] = edge_type
    normalized["edge_type"] = edge_type
    normalized["candidate_source"] = STRUCTURAL_CANDIDATE_SOURCE
    normalized["is_global"] = False
    normalized["score"] = float(edge.get("score") or 0.0)
    relation = compact_whitespace(edge.get("relation_text") or "")
    if not relation:
        relation = relation_text_for_edge(edge_type, src_node, dst_node)
    normalized["relation_text"] = relation
    return normalized


def pointer_candidate_order_key(edge: Dict[str, Any]) -> tuple[int, int, float, int]:
    edge_type = _edge_type(edge)
    return (
        1 if bool(edge.get("is_global")) else 0,
        EDGE_PRIORITY.get(edge_type, 99),
        -float(edge.get("score") or 0.0),
        int(edge["dst"]),
    )


def merge_pointer_candidates(
    *,
    structural_candidates: Sequence[Dict[str, Any]],
    pointer_candidate_k: int,
) -> List[Dict[str, Any]]:
    if pointer_candidate_k <= 0:
        raise ValueError(f"pointer_candidate_k must be positive, got {pointer_candidate_k}.")
    combined = [dict(edge) for edge in structural_candidates]
    if not combined:
        return []
    by_dst: Dict[int, Dict[str, Any]] = {}
    for edge in combined:
        dst = int(edge["dst"])
        current = by_dst.get(dst)
        if current is None or pointer_candidate_order_key(edge) < pointer_candidate_order_key(current):
            by_dst[dst] = dict(edge)
    return sorted(by_dst.values(), key=pointer_candidate_order_key)[:pointer_candidate_k]


def select_pointer_links(
    *,
    pointer_candidates: Sequence[Dict[str, Any]],
    pointer_link_k: int,
) -> List[Dict[str, Any]]:
    if pointer_link_k <= 0:
        raise ValueError(f"pointer_link_k must be positive, got {pointer_link_k}.")
    candidates = [dict(edge) for edge in pointer_candidates]
    if not candidates:
        return []
    candidates.sort(key=pointer_candidate_order_key)
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for edge in candidates:
        dst = int(edge["dst"])
        if dst in seen:
            continue
        seen.add(dst)
        chosen.append(edge)
        if len(chosen) >= pointer_link_k:
            return chosen[:pointer_link_k]
    return chosen[:pointer_link_k]


def flatten_pointer_links(node_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in node_summaries:
        src = int(item["node_id"])
        for link in item.get("pointer_links") or []:
            enriched = dict(link)
            enriched["src"] = src
            enriched["dst"] = int(link["dst"])
            enriched["type"] = str(link.get("type") or link.get("edge_type") or "")
            enriched["edge_type"] = str(link.get("edge_type") or link.get("type") or "")
            flattened.append(enriched)
    return flattened


def flatten_pointer_candidates(node_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in node_summaries:
        src = int(item["node_id"])
        for edge in item.get("pointer_candidates") or []:
            enriched = dict(edge)
            enriched["src"] = src
            enriched["dst"] = int(edge["dst"])
            enriched["type"] = str(edge.get("type") or edge.get("edge_type") or "")
            enriched["edge_type"] = str(edge.get("edge_type") or edge.get("type") or "")
            flattened.append(enriched)
    return flattened


def paper_summary_cache_key(record: Dict[str, Any]) -> str:
    paper_id = compact_whitespace(
        record.get("paper_id")
        or record.get("doc_meta", {}).get("paper_id")
        or ""
    )
    if not paper_id:
        record_id = str(record.get("id") or "").strip()
        if "::" in record_id:
            paper_id = record_id.split("::", 1)[0].strip()
    if paper_id:
        return paper_id
    return json.dumps(
        {
            "nodes": record.get("nodes") or [],
            "routing_links": record.get("routing_links") or [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_pointer_bundle(
    record: Dict[str, Any],
    *,
    structural_pointer_k: int,
    pointer_candidate_k: int,
    pointer_link_k: int,
) -> Dict[str, Any]:
    nodes = list(record.get("nodes") or [])
    if not nodes:
        raise ValueError(f"Record {record.get('id')} has no nodes.")
    node_summaries = [dict(item) for item in (record.get("node_summaries") or [])]
    if not node_summaries:
        raise ValueError(f"Record {record.get('id')} is missing node_summaries.")
    invalid_node_types = sorted(
        {
            str(node.get("node_type") or "")
            for node in nodes
            if str(node.get("node_type") or "") != "paragraph"
        }
    )
    if invalid_node_types:
        raise ValueError(
            f"Record {record.get('id')} contains non-paragraph nodes {invalid_node_types}. "
            "Rebuild the prepare artifacts with the paragraph-only graph."
        )

    node_index = node_lookup(nodes)
    summary_index: Dict[int, Dict[str, Any]] = {}
    for item in node_summaries:
        node_id = int(item["node_id"])
        if node_id in summary_index:
            raise ValueError(f"Record {record.get('id')} has duplicate node summary for node {node_id}.")
        if node_id not in node_index:
            raise ValueError(f"Record {record.get('id')} has node summary for unknown node {node_id}.")
        summary_item = dict(item)
        summary_item["node_id"] = node_id
        summary_index[node_id] = summary_item
    missing_summary_ids = sorted(set(node_index) - set(summary_index))
    if missing_summary_ids:
        raise ValueError(
            f"Record {record.get('id')} is missing node summaries for node ids {missing_summary_ids[:16]}."
        )

    structural_adjacency = build_routing_link_index(record, link_field="routing_links")

    structural_candidates_by_node: Dict[int, List[Dict[str, Any]]] = {}
    for node in nodes:
        node_id = int(node["node_id"])
        structural_candidates: List[Dict[str, Any]] = []
        for edge in structural_adjacency.get(node_id, []):
            dst = int(edge["dst"])
            dst_node = node_index.get(dst)
            if dst_node is None:
                continue
            structural_candidates.append(
                normalize_pointer_edge(
                    edge,
                    src_node=node,
                    dst_node=dst_node,
                )
            )
        structural_candidates_by_node[node_id] = structural_candidates[:structural_pointer_k]

    pointer_candidates_by_node: Dict[int, List[Dict[str, Any]]] = {}
    pointer_link_candidates_by_node: Dict[int, List[Dict[str, Any]]] = {}
    for node in nodes:
        node_id = int(node["node_id"])
        merged_candidates = merge_pointer_candidates(
            structural_candidates=structural_candidates_by_node[node_id],
            pointer_candidate_k=pointer_candidate_k,
        )
        if len(nodes) > 1 and not merged_candidates:
            raise ValueError(
                f"Pointer candidate construction produced no candidates for node {node_id} in record {record.get('id')}."
            )
        pointer_candidates_by_node[node_id] = merged_candidates
        pointer_link_candidates_by_node[node_id] = select_pointer_links(
            pointer_candidates=merged_candidates,
            pointer_link_k=pointer_link_k,
        )
        if len(nodes) > 1 and not pointer_link_candidates_by_node[node_id]:
            raise ValueError(
                f"Pointer link selection produced no retained links for node {node_id} in record {record.get('id')}."
            )

    finalized_summaries: List[Dict[str, Any]] = []
    for item in node_summaries:
        node_id = int(item["node_id"])
        pointer_candidates = list(pointer_candidates_by_node[node_id])
        candidates = list(pointer_link_candidates_by_node[node_id])
        if candidates:
            pointer_links: List[Dict[str, Any]] = []
            for candidate in candidates:
                enriched = dict(candidate)
                enriched["type"] = str(candidate.get("type") or candidate.get("edge_type") or "")
                enriched["edge_type"] = str(candidate.get("edge_type") or candidate.get("type") or "")
                enriched["link_tokens"] = []
                enriched["link_tokens_text"] = ""
                pointer_links.append(enriched)
        else:
            pointer_links = []

        enriched_item = dict(item)
        enriched_item.update(
            {
                "pointer_candidates": pointer_candidates,
                "pointer_links": pointer_links,
                "num_pointer_candidates": len(pointer_candidates),
                "num_pointer_links": len(pointer_links),
            }
        )
        finalized_summaries.append(enriched_item)

    flattened_pointer_candidates = flatten_pointer_candidates(finalized_summaries)
    flattened_pointer_links = flatten_pointer_links(finalized_summaries)
    return {
        "node_summaries": finalized_summaries,
        "pointer_candidates": flattened_pointer_candidates,
        "pointer_links": flattened_pointer_links,
    }


def apply_pointer_bundle(record: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    out["graph_schema_version"] = GRAPH_SCHEMA_VERSION
    out["node_summaries"] = copy.deepcopy(bundle["node_summaries"])
    out["pointer_candidates"] = copy.deepcopy(bundle["pointer_candidates"])
    out["pointer_links"] = copy.deepcopy(bundle["pointer_links"])
    meta = dict(record.get("meta") or {})
    meta["graph_schema_version"] = GRAPH_SCHEMA_VERSION
    meta["num_pointer_candidates"] = len(out["pointer_candidates"])
    meta["num_pointer_links"] = len(out["pointer_links"])
    out["meta"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pointer-links", type=int, default=4)
    parser.add_argument("--structural-pointer-k", type=int, default=4)
    parser.add_argument("--pointer-candidate-k", type=int, default=0)
    parser.add_argument("--pointer-link-k", type=int, default=0)
    parser.add_argument("--paper-summary-cache-size", type=int, default=1)
    args = parser.parse_args()

    if Path(args.data).resolve() == Path(args.out).resolve():
        raise ValueError("--data and --out must be different paths.")
    if int(args.structural_pointer_k) < 0:
        raise ValueError("--structural-pointer-k must be non-negative.")
    if int(args.paper_summary_cache_size) < 0:
        raise ValueError("--paper-summary-cache-size must be non-negative.")

    pointer_link_k = int(args.pointer_link_k) if int(args.pointer_link_k) > 0 else int(args.max_pointer_links)
    pointer_candidate_k = (
        int(args.pointer_candidate_k)
        if int(args.pointer_candidate_k) > 0
        else int(args.structural_pointer_k)
    )
    if pointer_link_k <= 0:
        raise ValueError("pointer_link_k must be positive. Set --pointer-link-k or --max-pointer-links to a positive value.")
    if pointer_candidate_k <= 0:
        raise ValueError("pointer_candidate_k must be positive. Configure structural pointer candidates.")
    if pointer_link_k > pointer_candidate_k:
        raise ValueError(
            f"pointer_link_k={pointer_link_k} cannot exceed pointer_candidate_k={pointer_candidate_k}."
        )

    wrote = 0
    paper_summary_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    paper_summary_cache_hits = 0
    paper_summary_cache_misses = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8", buffering=1) as f_out:
        iterator = tqdm(f_in, desc=f"pointer links {Path(args.data).name}", unit="records")
        for line in iterator:
            record = json.loads(line)
            pointer_bundle: Dict[str, Any] | None = None
            if int(args.paper_summary_cache_size) > 0:
                cache_key = paper_summary_cache_key(record)
                pointer_bundle = paper_summary_cache.get(cache_key)
                if pointer_bundle is not None:
                    paper_summary_cache.move_to_end(cache_key)
                    paper_summary_cache_hits += 1
                else:
                    paper_summary_cache_misses += 1
                    pointer_bundle = build_pointer_bundle(
                        record,
                        structural_pointer_k=args.structural_pointer_k,
                        pointer_candidate_k=pointer_candidate_k,
                        pointer_link_k=pointer_link_k,
                    )
                    paper_summary_cache[cache_key] = pointer_bundle
                    paper_summary_cache.move_to_end(cache_key)
                    while len(paper_summary_cache) > int(args.paper_summary_cache_size):
                        paper_summary_cache.popitem(last=False)
            enriched = apply_pointer_bundle(
                record,
                pointer_bundle
                if pointer_bundle is not None
                else build_pointer_bundle(
                    record,
                    structural_pointer_k=args.structural_pointer_k,
                    pointer_candidate_k=pointer_candidate_k,
                    pointer_link_k=pointer_link_k,
                ),
            )
            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            f_out.flush()
            wrote += 1
            iterator.set_postfix(
                wrote=wrote,
                cache_hits=paper_summary_cache_hits,
                cache_misses=paper_summary_cache_misses,
            )
    print(
        {
            "wrote": wrote,
            "cache_hits": paper_summary_cache_hits,
            "cache_misses": paper_summary_cache_misses,
        }
    )


if __name__ == "__main__":
    main()
