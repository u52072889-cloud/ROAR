#!/usr/bin/env python
"""Graph construction helpers for QASPER routing."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


EDGE_PRIORITY = {
    "paragraph_next": 0,
    "paragraph_prev": 1,
}

def node_lookup(nodes: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(node["node_id"]): dict(node) for node in nodes}


def relation_text_for_edge(edge_type: str, src: Dict[str, Any], dst: Dict[str, Any]) -> str:
    del src, dst
    if edge_type == "paragraph_next":
        return "next paragraph in document order"
    if edge_type == "paragraph_prev":
        return "previous paragraph in document order"
    return edge_type.replace("_", " ")


def make_edge(
    *,
    src: Dict[str, Any],
    dst: Dict[str, Any],
    edge_type: str,
    score: float | None = None,
) -> Dict[str, Any]:
    return {
        "src": int(src["node_id"]),
        "dst": int(dst["node_id"]),
        "edge_type": edge_type,
        "score": float(score) if score is not None else None,
        "relation_text": relation_text_for_edge(edge_type, src, dst),
    }


def build_structural_edges(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    invalid_node_types = sorted(
        {
            str(node.get("node_type") or "")
            for node in nodes
            if str(node.get("node_type") or "") != "paragraph"
        }
    )
    if invalid_node_types:
        raise ValueError(
            f"Paragraph-only graph expected only paragraph nodes, but found node types {invalid_node_types}."
        )
    ordered_paragraphs = sorted(
        (dict(node) for node in nodes if node.get("node_type") == "paragraph"),
        key=lambda item: (int(item.get("section_index", -1)), int(item.get("paragraph_index", -1)), int(item["node_id"])),
    )
    edges: List[Dict[str, Any]] = []
    for left, right in zip(ordered_paragraphs[:-1], ordered_paragraphs[1:]):
        edges.append(make_edge(src=left, dst=right, edge_type="paragraph_next", score=1.0))
        edges.append(make_edge(src=right, dst=left, edge_type="paragraph_prev", score=1.0))
    return edges


def dedupe_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for edge in edges:
        key = (int(edge["src"]), int(edge["dst"]), str(edge["edge_type"]))
        current = best.get(key)
        if current is None:
            best[key] = dict(edge)
            continue
        score = edge.get("score")
        current_score = current.get("score")
        edge_score = float(score) if score is not None else -math.inf
        current_score_value = float(current_score) if current_score is not None else -math.inf
        if edge_score > current_score_value:
            best[key] = dict(edge)
    return sorted(best.values(), key=lambda item: (int(item["src"]), EDGE_PRIORITY.get(str(item["edge_type"]), 99), int(item["dst"])))


def build_routing_link_index(
    record: Dict[str, Any],
    *,
    link_field: str = "routing_links",
) -> Dict[int, List[Dict[str, Any]]]:
    adjacency: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for edge in record.get(link_field, []) or []:
        adjacency[int(edge["src"])].append(dict(edge))
    for src in adjacency:
        adjacency[src] = sorted(
            adjacency[src],
            key=lambda edge: (
                EDGE_PRIORITY.get(str(edge.get("edge_type") or edge.get("type") or ""), 99),
                -float(edge.get("score") or 0.0),
                int(edge["dst"]),
            ),
        )
    return adjacency

