#!/usr/bin/env python
"""Train the final graph-routed QASPER reader."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from data_utils import (
    SPECIAL_TOKENS,
    compact_whitespace,
    effective_model_max_length,
    resolve_local_hf_model_path,
    set_offline_hf_env,
    strict_encode,
    strict_special_token_id,
)
from graph_utils import build_routing_link_index
from qasper_eval import aggregate_prediction_metrics, best_answer_scores, best_evidence_scores, qasper_question_key


DEFAULT_GRAPH_HEAD_HIDDEN = 256
DEFAULT_GRAPH_HEAD_DROPOUT = 0.1
DEFAULT_MAX_HOPS = 4
DEFAULT_BEAM_WIDTH = 6
DEFAULT_TOP_K = 16
DEFAULT_MAX_EVIDENCE_NODES = 8
DEFAULT_ANSWER_ROUTER_BASELINE_MOMENTUM = 0.95
CURRENT_PIPELINE_FIXED_SEED_EVIDENCE_COUNT = 5
CURRENT_PIPELINE_FRONTIER_SELECT_COUNT = 3
LOCAL_EDGE_TYPES = {"paragraph_next", "paragraph_prev"}
SEED_TAIL_SCORE_PENALTY = 0.25


def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


class MemoryVocab:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for memory encoding.")

    @staticmethod
    def _normalize_token(raw_token: str) -> str:
        token = str(raw_token or "").strip()
        if token.startswith("MEM_"):
            token = token[4:]
        elif token.startswith("REL_"):
            token = token[4:]
        return token.replace("_", " ").strip()

    def encode(self, tokens: Sequence[str]) -> List[int]:
        normalized = [self._normalize_token(token) for token in (tokens or [])]
        normalized = [token for token in normalized if token]
        if not normalized:
            raise ValueError("Memory token sequence is empty. Rebuild compact memory artifacts with non-empty node/link tokens.")
        return strict_encode(
            self.tokenizer,
            " ; ".join(normalized),
            add_special_tokens=False,
            context="memory token sequence",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "lm_tokenizer",
            "pad_id": int(self.pad_id),
        }


def build_memory_vocab(tokenizer: Any) -> MemoryVocab:
    return MemoryVocab(tokenizer)


def split_node_sum_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for part in str(text or "").replace("\n", ";").split(";"):
        phrase = compact_whitespace(part)
        if phrase:
            phrases.append(phrase)
    return phrases


def node_memory_phrases(summary: Dict[str, Any]) -> List[str]:
    phrases: List[str] = []
    section_title = compact_whitespace(summary.get("section_title") or "")
    if section_title:
        phrases.append(section_title)
    node_sum_text = compact_whitespace(summary.get("node_sum_text") or "")
    if not node_sum_text:
        raise ValueError(f"Node summary {summary.get('node_id')} is missing node_sum_text.")
    phrases.extend(split_node_sum_phrases(node_sum_text))
    return phrases


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    total = (hidden * weights).sum(dim=-2)
    denom = weights.sum(dim=-2).clamp_min(1.0)
    return total / denom


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class RoutedReaderDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Any,
        memory_vocab: MemoryVocab,
        *,
        max_len: int,
        max_hops: int,
        beam_width: int,
        top_k: int,
        max_evidence_nodes: int,
        max_records: int = 0,
        include_summary_header: bool = False,
        route_seed_limit: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.memory_vocab = memory_vocab
        self.max_len = int(max_len)
        self.max_hops = int(max_hops)
        self.beam_width = int(beam_width)
        self.top_k = int(top_k)
        self.max_evidence_nodes = int(max_evidence_nodes)
        self.include_summary_header = bool(include_summary_header)
        self.route_seed_limit = int(route_seed_limit)
        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
                if max_records and len(self.records) >= max_records:
                    break

    def __len__(self) -> int:
        return len(self.records)

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

    def _render_target(self, record: Dict[str, Any]) -> str:
        return "<target>" + str(record.get("target_answer", {}).get("normalized_answer") or "") + "</target>"

    def _render_evidence_segment(self, node: Dict[str, Any], node_memory: Sequence[str]) -> str:
        title = str(node.get("section_title") or "").strip()
        text = str(node.get("text") or "")
        parts: List[str] = []
        if self.include_summary_header:
            parts.append("<summary>" + "; ".join(str(phrase) for phrase in node_memory) + "</summary>")
        evidence_text = f"Section: {title}\nParagraph {int(node['node_id'])}\n{text}"
        parts.append("<evidence>" + evidence_text + "</evidence>")
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _ordered_paragraph_nodes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        invalid_node_types = sorted(
            {
                str(node.get("node_type") or "")
                for node in (record.get("nodes") or [])
                if str(node.get("node_type") or "") != "paragraph"
            }
        )
        if invalid_node_types:
            raise ValueError(
                f"Record {record.get('id')} contains non-paragraph nodes {invalid_node_types}. "
                "The final routed reader requires a paragraph-only graph."
            )
        return sorted(
            (dict(node) for node in (record.get("nodes") or [])),
            key=lambda node: (
                int(node.get("section_index") or 0),
                int(node.get("paragraph_index") or 0),
                int(node["node_id"]),
            ),
        )

    def build_example(self, record: Dict[str, Any], *, include_target: bool) -> Dict[str, Any]:
        record_id = str(record.get("id") or "")
        summary_index = {
            int(item["node_id"]): dict(item)
            for item in (record.get("node_summaries") or [])
        }
        if not summary_index:
            raise ValueError(
                f"Record {record_id} is missing node_summaries. "
                "Train the final routed reader on the summary+link artifact, not the raw graph artifact."
            )
        prefix_ids = self._encode(self._render_prefix(record), context=f"prefix for {record_id}")
        query_ids = self._encode(self._render_query(record), context=f"query for {record_id}")
        target_ids = self._encode(self._render_target(record), context=f"target for {record_id}") if include_target else []

        ordered_nodes = self._ordered_paragraph_nodes(record)
        node_ids: List[int] = []
        node_sum_token_ids: List[List[int]] = []
        evidence_input_ids: List[List[int]] = []
        for node in ordered_nodes:
            node_id = int(node["node_id"])
            summary = summary_index.get(node_id)
            if summary is None:
                raise ValueError(f"Record {record_id} is missing node summary for node_id={node_id}.")
            node_memory = node_memory_phrases(summary)
            if not node_memory:
                raise ValueError(
                    f"Record {record_id} node {node_id} has empty node_sum_text. "
                    "Compact memory must preserve non-empty node summary phrases."
                )
            node_ids.append(node_id)
            node_sum_token_ids.append(self.memory_vocab.encode(node_memory))
            evidence_input_ids.append(
                self._encode(
                    self._render_evidence_segment(node, node_memory),
                    context=f"evidence node {node_id} for {record_id}",
                )
            )

        node_index_by_id = {node_id: idx for idx, node_id in enumerate(node_ids)}
        adjacency = build_routing_link_index(record, link_field="pointer_links")
        edge_src_index: List[int] = []
        edge_dst_index: List[int] = []
        edge_link_token_ids: List[List[int]] = []
        edge_types: List[str] = []
        edge_link_tokens: List[List[str]] = []
        for src_id in node_ids:
            for edge in adjacency.get(int(src_id), []):
                dst_id = int(edge["dst"])
                if dst_id not in node_index_by_id:
                    continue
                edge_type = str(edge.get("edge_type") or edge.get("type") or "")
                if edge_type not in LOCAL_EDGE_TYPES:
                    continue
                link_tokens = list(edge.get("link_tokens") or [])
                if not link_tokens:
                    raise ValueError(
                        f"Record {record_id} edge {int(src_id)}->{dst_id} has empty link_tokens. "
                        "Run build_qasper_link_summaries.py and train on the linksum artifact."
                    )
                edge_src_index.append(node_index_by_id[int(src_id)])
                edge_dst_index.append(node_index_by_id[dst_id])
                edge_link_token_ids.append(self.memory_vocab.encode(link_tokens))
                edge_types.append(edge_type)
                edge_link_tokens.append([str(token) for token in link_tokens])
        if not edge_src_index:
            raise ValueError(f"Record {record_id} has no routable pointer_links after compact memory filtering.")
        query = dict(record.get("query") or {})

        node_retrieval_scores = [0.0] * len(node_ids)
        node_score_items = list(query.get("node_scores") or [])
        if not node_score_items:
            raise ValueError(
                f"Record {record_id} is missing query.node_scores. "
                "Run recompute_qasper_embedding_seed_candidates.py before answer-router training."
            )
        if node_score_items:
            retrieval_score_by_node: Dict[int, float] = {}
            for item in node_score_items:
                node_id = int(item["node_id"])
                if "score" in item:
                    retrieval_score_by_node[node_id] = float(item["score"])
                else:
                    raise ValueError(f"Record {record_id} query.node_scores has an item without score.")
            missing_scores = [int(node_id) for node_id in node_ids if int(node_id) not in retrieval_score_by_node]
            if missing_scores:
                raise ValueError(
                    f"Record {record_id} is missing query.node_scores for node_ids={missing_scores[:8]}."
                )
            node_retrieval_scores = [
                float(retrieval_score_by_node.get(int(node_id), 0.0))
                for node_id in node_ids
            ]

        raw_seed_candidates = list(query.get("node_scores") or [])
        if not raw_seed_candidates:
            raise ValueError(
                f"Record {record_id} has no ranked seed candidates. "
                "The answer router requires query.node_scores."
            )
        seed_limit = int(self.route_seed_limit) if int(self.route_seed_limit) > 0 else len(raw_seed_candidates)
        seed_candidate_indices = [
            node_index_by_id[int(item["node_id"])]
            for item in raw_seed_candidates[:seed_limit]
            if int(item["node_id"]) in node_index_by_id
        ]
        if not seed_candidate_indices:
            raise ValueError(f"Record {record_id} is missing usable query.node_scores.")

        static_efficiency = {
            "prefix_tokens": float(len(prefix_ids)),
            "query_tokens": float(len(query_ids)),
            "target_tokens": float(len(target_ids)),
            "graph_nodes": float(len(node_ids)),
            "graph_edges": float(len(edge_src_index)),
        }
        return {
            "record_id": record_id,
            "prefix_input_ids": prefix_ids,
            "query_input_ids": query_ids,
            "target_input_ids": target_ids,
            "node_ids": node_ids,
            "node_sum_token_ids": node_sum_token_ids,
            "evidence_input_ids": evidence_input_ids,
            "node_retrieval_scores": node_retrieval_scores,
            "edge_src_index": edge_src_index,
            "edge_dst_index": edge_dst_index,
            "edge_link_token_ids": edge_link_token_ids,
            "edge_types": edge_types,
            "edge_link_tokens": edge_link_tokens,
            "seed_candidate_indices": seed_candidate_indices,
            "efficiency_stats": static_efficiency,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.build_example(self.records[idx], include_target=True)

    def generation_item(self, idx: int) -> Dict[str, Any]:
        return self.build_example(self.records[idx], include_target=False)


@dataclass
class RoutedReaderCollator:
    tokenizer: Any

    @staticmethod
    def _pad_matrix(values: List[List[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((len(item) for item in values), default=0)
        if max_len <= 0:
            return (
                torch.zeros((len(values), 0), dtype=torch.long),
                torch.zeros((len(values), 0), dtype=torch.bool),
            )
        padded = []
        mask = []
        for item in values:
            pad_len = max_len - len(item)
            padded.append(list(item) + [pad_id] * pad_len)
            mask.append([1] * len(item) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("The final routed reader currently requires --batch 1.")
        example = batch[0]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        prefix_ids, prefix_mask = self._pad_matrix([example["prefix_input_ids"]], pad_id)
        query_ids, query_mask = self._pad_matrix([example["query_input_ids"]], pad_id)
        target_ids, target_mask = self._pad_matrix([example["target_input_ids"]], pad_id)
        node_sum_ids, node_sum_mask = self._pad_matrix(example["node_sum_token_ids"], pad_id)
        evidence_ids, evidence_mask = self._pad_matrix(example["evidence_input_ids"], pad_id)
        edge_link_ids, edge_link_mask = self._pad_matrix(example["edge_link_token_ids"], pad_id)

        return {
            "record_id": example["record_id"],
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": prefix_mask,
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "target_input_ids": target_ids,
            "target_attention_mask": target_mask,
            "node_ids": torch.tensor([example["node_ids"]], dtype=torch.long),
            "node_sum_token_ids": node_sum_ids.unsqueeze(0),
            "node_sum_token_mask": node_sum_mask.unsqueeze(0),
            "evidence_input_ids": evidence_ids.unsqueeze(0),
            "evidence_attention_mask": evidence_mask.unsqueeze(0),
            "node_retrieval_scores": torch.tensor([example["node_retrieval_scores"]], dtype=torch.float32),
            "edge_src_index": torch.tensor([example["edge_src_index"]], dtype=torch.long),
            "edge_dst_index": torch.tensor([example["edge_dst_index"]], dtype=torch.long),
            "edge_link_token_ids": edge_link_ids.unsqueeze(0),
            "edge_link_token_mask": edge_link_mask.unsqueeze(0),
            "seed_candidate_indices": torch.tensor([example["seed_candidate_indices"]], dtype=torch.long),
            "edge_types": [list(example["edge_types"])],
            "edge_link_tokens": [list(example["edge_link_tokens"])],
            "efficiency_stats": [dict(example["efficiency_stats"])],
        }


class RoutedReaderModel(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        memory_vocab: MemoryVocab,
        max_len: int,
        top_k: int,
        max_hops: int,
        beam_width: int,
        max_evidence_nodes: int,
        hidden_dim: int,
        dropout: float,
        answer_router_loss_weight: float,
        answer_router_baseline_momentum: float,
        answer_router_entropy_weight: float,
        router_start_prior_weight: float,
        router_edge_prior_weight: float,
        fixed_seed_evidence_count: int,
        frontier_select_count: int,
        frontier_hop_count: int,
        frontier_baseline_seed_count: int,
        frontier_max_per_seed: int,
        query_conditioned_frontier_reranker: bool,
        frontier_query_reranker_weight: float,
        use_seed_evidence_only: bool,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.max_len = int(max_len)
        self.top_k = int(top_k)
        self.max_hops = int(max_hops)
        self.beam_width = int(beam_width)
        self.max_evidence_nodes = int(max_evidence_nodes)
        self.answer_router_loss_weight = float(answer_router_loss_weight)
        self.answer_router_baseline_momentum = float(answer_router_baseline_momentum)
        self.answer_router_entropy_weight = float(answer_router_entropy_weight)
        self.router_start_prior_weight = float(router_start_prior_weight)
        self.router_edge_prior_weight = float(router_edge_prior_weight)
        self.fixed_seed_evidence_count = int(fixed_seed_evidence_count)
        self.frontier_select_count = int(frontier_select_count)
        self.frontier_hop_count = int(frontier_hop_count)
        self.frontier_baseline_seed_count = int(frontier_baseline_seed_count)
        self.frontier_max_per_seed = int(frontier_max_per_seed)
        self.query_conditioned_frontier_reranker = bool(query_conditioned_frontier_reranker)
        self.frontier_query_reranker_weight = float(frontier_query_reranker_weight)
        self.use_seed_evidence_only = bool(use_seed_evidence_only)
        lm_hidden = int(base_model.get_input_embeddings().embedding_dim)

        del memory_vocab
        self.memory_proj = nn.Linear(lm_hidden, hidden_dim)
        router_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.router_encoder = nn.TransformerEncoder(router_encoder_layer, num_layers=2)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_message = mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.node_update = nn.LayerNorm(hidden_dim)
        self.route_init = mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.route_state_update = nn.GRUCell(hidden_dim * 3, hidden_dim)
        self.route_state_norm = nn.LayerNorm(hidden_dim)
        self.start_head = mlp(hidden_dim * 3, hidden_dim, 1, dropout)
        self.next_hop_head = mlp(hidden_dim * 5, hidden_dim, 1, dropout)
        self.dropout = nn.Dropout(dropout)

        self._last_route_result: Dict[str, Any] = {}
        self._last_evidence_node_ids: List[int] = []
        self._last_generation_prompt_len: int = 0
        self._last_efficiency_stats: Dict[str, float] = {}
        self._last_training_metrics: Dict[str, float] = {}
        self._answer_loss_baseline: float | None = None

    def save_pretrained(self, save_directory: str, *args, **kwargs) -> None:
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        torch.save(
            {
                "router_state_dict": {
                    "memory_proj": self.memory_proj.state_dict(),
                    "router_encoder": self.router_encoder.state_dict(),
                    "memory_norm": self.memory_norm.state_dict(),
                    "query_proj": self.query_proj.state_dict(),
                    "text_proj": self.text_proj.state_dict(),
                    "edge_message": self.edge_message.state_dict(),
                    "node_update": self.node_update.state_dict(),
                    "route_init": self.route_init.state_dict(),
                    "route_state_update": self.route_state_update.state_dict(),
                    "route_state_norm": self.route_state_norm.state_dict(),
                    "start_head": self.start_head.state_dict(),
                    "next_hop_head": self.next_hop_head.state_dict(),
                },
                "config": {
                    "max_len": self.max_len,
                    "top_k": self.top_k,
                    "max_hops": self.max_hops,
                    "beam_width": self.beam_width,
                    "max_evidence_nodes": self.max_evidence_nodes,
                    "answer_router_loss_weight": self.answer_router_loss_weight,
                    "answer_router_baseline_momentum": self.answer_router_baseline_momentum,
                    "answer_router_entropy_weight": self.answer_router_entropy_weight,
                    "router_start_prior_weight": self.router_start_prior_weight,
                    "router_edge_prior_weight": self.router_edge_prior_weight,
                    "fixed_seed_evidence_count": self.fixed_seed_evidence_count,
                    "frontier_select_count": self.frontier_select_count,
                    "frontier_hop_count": self.frontier_hop_count,
                    "frontier_baseline_seed_count": self.frontier_baseline_seed_count,
                    "frontier_max_per_seed": self.frontier_max_per_seed,
                    "query_conditioned_frontier_reranker": self.query_conditioned_frontier_reranker,
                    "frontier_query_reranker_weight": self.frontier_query_reranker_weight,
                    "use_seed_evidence_only": self.use_seed_evidence_only,
                },
            },
            Path(save_directory) / "graph_router_aux.pt",
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        return self.base_model.gradient_checkpointing_disable()

    def _lm_mean(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embed = self.base_model.get_input_embeddings()(input_ids).detach()
        hidden = self.memory_proj(embed.to(dtype=self.memory_proj.weight.dtype))
        encoded = self.router_encoder(
            hidden,
            src_key_padding_mask=~attention_mask.bool(),
        )
        return self.memory_norm(masked_mean_pool(encoded, attention_mask))

    def _memory_mean(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._lm_mean(input_ids, attention_mask)

    def _encode_router(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        node_sum_token_ids: torch.Tensor,
        node_sum_token_mask: torch.Tensor,
        edge_src_index: torch.Tensor,
        edge_dst_index: torch.Tensor,
        edge_link_token_ids: torch.Tensor,
        edge_link_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(query_input_ids.shape[0]) != 1:
            raise ValueError("The final routed reader currently supports batch_size=1 only.")

        query_repr = self.query_proj(
            self._lm_mean(query_input_ids, query_attention_mask).to(dtype=self.query_proj.weight.dtype)
        )
        node_repr = self._memory_mean(node_sum_token_ids[0], node_sum_token_mask[0])
        link_repr = self._memory_mean(edge_link_token_ids[0], edge_link_token_mask[0])
        src_repr = node_repr[edge_src_index[0]]
        dst_repr = node_repr[edge_dst_index[0]]

        messages = self.edge_message(torch.cat([src_repr, link_repr, dst_repr], dim=-1))
        aggregated = torch.zeros_like(node_repr)
        counts = torch.zeros((node_repr.shape[0], 1), dtype=node_repr.dtype, device=node_repr.device)
        aggregated.index_add_(0, edge_dst_index[0], messages.to(dtype=aggregated.dtype))
        counts.index_add_(
            0,
            edge_dst_index[0],
            torch.ones((messages.shape[0], 1), dtype=node_repr.dtype, device=node_repr.device),
        )
        node_repr = self.node_update(node_repr + aggregated / counts.clamp_min(1.0))

        expanded_query_nodes = query_repr.expand(node_repr.shape[0], -1)
        start_logits = self.start_head(
            torch.cat([expanded_query_nodes, node_repr, expanded_query_nodes * node_repr], dim=-1)
        ).squeeze(-1)

        src_repr = node_repr[edge_src_index[0]]
        dst_repr = node_repr[edge_dst_index[0]]
        expanded_query_edges = query_repr.expand(link_repr.shape[0], -1)
        edge_logits = self.next_hop_head(
            torch.cat(
                [
                    expanded_query_edges,
                    src_repr,
                    link_repr,
                    dst_repr,
                    expanded_query_edges * dst_repr,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return query_repr.squeeze(0), node_repr, link_repr, start_logits, edge_logits

    def _normalized_retrieval_scores(
        self,
        node_retrieval_scores: torch.Tensor | None,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if node_retrieval_scores is None or int(node_retrieval_scores.numel()) <= 0:
            return torch.zeros_like(like)
        scores = node_retrieval_scores.to(device=like.device, dtype=like.dtype).view(-1)
        if int(scores.shape[0]) != int(like.shape[0]):
            raise ValueError(
                f"node_retrieval_scores has {int(scores.shape[0])} items, expected {int(like.shape[0])}."
            )
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError("node_retrieval_scores contains non-finite values.")
        score_min = scores.min()
        score_max = scores.max()
        denom = (score_max - score_min).clamp_min(1e-6)
        return (scores - score_min) / denom

    @staticmethod
    def _valid_seed_indices(
        *,
        seed_candidate_indices: torch.Tensor,
        num_nodes: int,
    ) -> List[int]:
        valid: List[int] = []
        seen = set()
        for raw_idx in seed_candidate_indices.tolist():
            node_idx = int(raw_idx)
            if node_idx < 0 or node_idx >= int(num_nodes) or node_idx in seen:
                continue
            seen.add(node_idx)
            valid.append(node_idx)
        if not valid:
            raise ValueError("Router received no valid seed candidates.")
        return valid

    def _frontier_candidates(
        self,
        *,
        base_indices: Sequence[int],
        start_logits: torch.Tensor,
        edge_src_index: torch.Tensor,
        edge_dst_index: torch.Tensor,
        edge_types: Sequence[str],
        edge_link_tokens: Sequence[Sequence[str]],
        edge_logits: torch.Tensor,
        node_retrieval_scores: torch.Tensor | None,
        like: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        node_prior_scores = self._normalized_retrieval_scores(node_retrieval_scores, like=like)
        edge_prior_weight = float(self.router_edge_prior_weight)
        outgoing: Dict[int, List[int]] = {}
        for edge_idx, src_idx in enumerate(edge_src_index.tolist()):
            outgoing.setdefault(int(src_idx), []).append(int(edge_idx))

        base_set = {int(idx) for idx in base_indices}
        zero = edge_logits.new_zeros(())
        states: List[Dict[str, Any]] = [
            {
                "current": int(idx),
                "root": int(idx),
                "line": [int(idx)],
                "path_edges": [],
                "score": zero,
                "path_score": zero,
            }
            for idx in base_indices
        ]
        best_by_dst: Dict[int, Dict[str, Any]] = {}
        max_hops = max(1, int(self.frontier_hop_count))
        for depth in range(1, max_hops + 1):
            next_states: List[Dict[str, Any]] = []
            for state in states:
                current_idx = int(state["current"])
                for edge_idx in outgoing.get(current_idx, []):
                    dst_idx = int(edge_dst_index[int(edge_idx)].item())
                    if dst_idx in state["line"]:
                        continue
                    step_score = edge_logits[int(edge_idx)] + edge_prior_weight * node_prior_scores[dst_idx]
                    path_score = state["path_score"] + step_score
                    path_edges = list(state["path_edges"]) + [int(edge_idx)]
                    line = list(state["line"]) + [dst_idx]
                    candidate_state = {
                        "current": dst_idx,
                        "root": int(state["root"]),
                        "line": line,
                        "path_edges": path_edges,
                        "score": path_score,
                        "path_score": path_score,
                        "depth": int(depth),
                    }
                    next_states.append(candidate_state)
                    if dst_idx in base_set:
                        continue
                    previous = best_by_dst.get(dst_idx)
                    candidate_score = start_logits[dst_idx] + path_score / float(depth)
                    if self.query_conditioned_frontier_reranker:
                        candidate_score = (
                            candidate_score
                            + float(self.frontier_query_reranker_weight) * node_prior_scores[dst_idx]
                        )
                    if previous is None or float(candidate_score.detach().item()) > float(previous["score"].detach().item()):
                        candidate_state["score"] = candidate_score
                        candidate_state["path_score"] = path_score
                        candidate_state["candidate_source"] = "frontier"
                        best_by_dst[dst_idx] = candidate_state
            states = next_states

        candidates = list(best_by_dst.values())
        candidates.sort(
            key=lambda item: (
                -float(item["score"].detach().item()),
                int(item["depth"]),
                int(item["current"]),
            )
        )
        for candidate in candidates:
            selected_edges: List[Dict[str, Any]] = []
            for edge_idx in candidate["path_edges"]:
                src_idx = int(edge_src_index[int(edge_idx)].item())
                dst_idx = int(edge_dst_index[int(edge_idx)].item())
                selected_edges.append(
                    {
                        "src_idx": src_idx,
                        "dst_idx": dst_idx,
                        "edge_idx": int(edge_idx),
                        "edge_type": str(edge_types[int(edge_idx)]),
                        "link_tokens": list(edge_link_tokens[int(edge_idx)]),
                    }
                )
            candidate["selected_edges"] = selected_edges
        return candidates

    def _seed_root_candidates(
        self,
        *,
        root_indices: Sequence[int],
        start_logits: torch.Tensor,
        node_retrieval_scores: torch.Tensor | None,
        like: torch.Tensor,
        candidate_source: str = "seed_root",
    ) -> List[Dict[str, Any]]:
        node_prior_scores = self._normalized_retrieval_scores(node_retrieval_scores, like=like)
        start_prior_weight = float(self.router_start_prior_weight)
        candidates: List[Dict[str, Any]] = []
        for idx in root_indices:
            node_idx = int(idx)
            query_prior_weight = (
                float(self.frontier_query_reranker_weight)
                if self.query_conditioned_frontier_reranker
                else 0.0
            )
            candidates.append(
                {
                    "current": node_idx,
                    "root": node_idx,
                    "line": [node_idx],
                    "path_edges": [],
                    "selected_edges": [],
                    "score": start_logits[node_idx]
                    + (start_prior_weight + query_prior_weight) * node_prior_scores[node_idx],
                    "depth": 0,
                    "candidate_source": str(candidate_source),
                }
            )
        return candidates

    def _frontier_route(
        self,
        *,
        node_ids: torch.Tensor,
        seed_candidate_indices: torch.Tensor,
        node_retrieval_scores: torch.Tensor,
        edge_src_index: torch.Tensor,
        edge_dst_index: torch.Tensor,
        edge_types: Sequence[str],
        edge_link_tokens: Sequence[Sequence[str]],
        start_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        sample: bool,
    ) -> Dict[str, Any]:
        node_ids_list = [int(node_id) for node_id in node_ids.tolist()]
        valid_seed_indices = self._valid_seed_indices(
            seed_candidate_indices=seed_candidate_indices,
            num_nodes=len(node_ids_list),
        )
        fixed_seed_budget = min(int(self.fixed_seed_evidence_count), len(valid_seed_indices))
        fixed_seed_indices = list(valid_seed_indices[:fixed_seed_budget])
        root_indices = list(fixed_seed_indices)
        route_budget = min(
            max(0, int(self.frontier_select_count)),
            max(0, len(node_ids_list) - len(fixed_seed_indices)),
        )
        selection_budget = min(
            int(self.top_k),
            len(fixed_seed_indices) + route_budget,
        )
        seed_tail_indices = list(valid_seed_indices[fixed_seed_budget:])
        candidates = self._frontier_candidates(
            base_indices=root_indices,
            start_logits=start_logits,
            edge_src_index=edge_src_index,
            edge_dst_index=edge_dst_index,
            edge_types=edge_types,
            edge_link_tokens=edge_link_tokens,
            edge_logits=edge_logits,
            node_retrieval_scores=node_retrieval_scores,
            like=torch.zeros((len(node_ids_list),), dtype=edge_logits.dtype, device=edge_logits.device),
        )
        candidates.extend(
            self._seed_root_candidates(
                root_indices=seed_tail_indices,
                start_logits=start_logits,
                node_retrieval_scores=node_retrieval_scores,
                like=torch.zeros((len(node_ids_list),), dtype=edge_logits.dtype, device=edge_logits.device),
                candidate_source="seed_tail",
            )
        )
        frontier_candidate_count = sum(
            1 for item in candidates if str(item.get("candidate_source") or "") == "frontier"
        )
        seed_tail_candidate_count = sum(
            1 for item in candidates if str(item.get("candidate_source") or "") == "seed_tail"
        )
        best_by_current: Dict[int, Dict[str, Any]] = {}
        for candidate in candidates:
            current = int(candidate["current"])
            previous = best_by_current.get(current)
            if previous is None or float(candidate["score"].detach().item()) > float(previous["score"].detach().item()):
                best_by_current[current] = candidate
        candidates = list(best_by_current.values())
        candidates.sort(
            key=lambda item: (
                -float(item["score"].detach().item()),
                int(item["depth"]),
                int(item["current"]),
            )
        )
        selected_frontier: List[Dict[str, Any]] = []
        selected_frontier_nodes = set(fixed_seed_indices)
        log_prob_terms: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        root_counts: Dict[int, int] = {}
        max_per_seed = int(self.frontier_max_per_seed)
        for _ in range(route_budget):
            available = [
                item
                for item in candidates
                if int(item["current"]) not in selected_frontier_nodes
                and (
                    str(item.get("candidate_source") or "") == "seed_root"
                    or max_per_seed <= 0
                    or root_counts.get(int(item["root"]), 0) < max_per_seed
                )
            ]
            if not available:
                break
            available_scores = [item["score"] for item in available]
            if sample:
                logits = torch.stack(available_scores)
                dist = torch.distributions.Categorical(logits=logits)
                sampled_pos = dist.sample()
                selected = dict(available[int(sampled_pos.item())])
                selected["selection_score"] = available_scores[int(sampled_pos.item())]
                log_prob_terms.append(dist.log_prob(sampled_pos))
                entropy_terms.append(dist.entropy())
            else:
                best_pos = max(
                    range(len(available)),
                    key=lambda idx: (
                        float(available_scores[idx].detach().item()),
                        -int(available[idx]["depth"]),
                        -int(available[idx]["current"]),
                    ),
                )
                selected = dict(available[best_pos])
                selected["selection_score"] = available_scores[best_pos]
            selected_frontier.append(selected)
            selected_frontier_nodes.add(int(selected["current"]))
            if str(selected.get("candidate_source") or "") == "frontier":
                root = int(selected["root"])
                root_counts[root] = root_counts.get(root, 0) + 1

        selected_node_indices: List[int] = list(fixed_seed_indices)
        selected_seen = set(fixed_seed_indices)
        selected_edges: List[Dict[str, Any]] = []
        emitted_edges = set()
        for item in selected_frontier:
            dst_idx = int(item["current"])
            if dst_idx not in selected_seen:
                selected_seen.add(dst_idx)
                selected_node_indices.append(dst_idx)
            for edge in item.get("selected_edges") or []:
                edge_idx = int(edge["edge_idx"])
                if edge_idx in emitted_edges:
                    continue
                emitted_edges.add(edge_idx)
                selected_edges.append(
                    {
                        "src": int(node_ids_list[int(edge["src_idx"])]),
                        "dst": int(node_ids_list[int(edge["dst_idx"])]),
                        "edge_type": str(edge["edge_type"]),
                        "routing_score": float(item.get("selection_score", item["score"]).detach().item()),
                        "link_tokens": list(edge["link_tokens"]),
                        "root_node_id": int(node_ids_list[int(item["root"])]),
                        "round": int(item["depth"]),
                    }
                )

        result: Dict[str, Any] = {
            "selected_node_indices": selected_node_indices[: selection_budget],
            "selected_node_ids": [node_ids_list[idx] for idx in selected_node_indices[:selection_budget]],
            "start_node_ids": [node_ids_list[idx] for idx in root_indices],
            "selected_edges": selected_edges,
            "hops_used": len(selected_edges),
            "frontier_candidate_count": frontier_candidate_count,
            "frontier_selected_count": len(selected_frontier),
            "frontier_path_selected_count": sum(
                1 for item in selected_frontier if str(item.get("candidate_source") or "") == "frontier"
            ),
            "seed_tail_candidate_count": seed_tail_candidate_count,
            "seed_tail_selected_count": sum(
                1 for item in selected_frontier if str(item.get("candidate_source") or "") == "seed_tail"
            ),
            "fixed_seed_count": len(fixed_seed_indices),
        }
        if sample:
            if log_prob_terms:
                result["route_log_prob"] = torch.stack(log_prob_terms).sum()
                result["route_entropy"] = torch.stack(entropy_terms).sum()
            else:
                result["route_log_prob"] = edge_logits.new_zeros(())
                result["route_entropy"] = edge_logits.new_zeros(())
        return result

    def _reader_node_indices(
        self,
        *,
        route_result: Dict[str, Any],
    ) -> List[int]:
        routed: List[int] = []
        seen = set()
        for index in route_result.get("selected_node_indices") or []:
            idx = int(index)
            if idx in seen:
                continue
            seen.add(idx)
            routed.append(idx)
        reader_indices = routed[: self.max_evidence_nodes]
        reader_indices.sort()
        if not reader_indices:
            raise ValueError("Reader received no routed evidence nodes.")
        return reader_indices

    def _seed_reader_node_indices(
        self,
        *,
        seed_candidate_indices: torch.Tensor,
        num_nodes: int,
    ) -> List[int]:
        reader_indices = self._valid_seed_indices(
            seed_candidate_indices=seed_candidate_indices,
            num_nodes=num_nodes,
        )[: self.max_evidence_nodes]
        reader_indices.sort()
        if not reader_indices:
            raise ValueError("Reader received no seed evidence nodes.")
        return reader_indices

    def _seed_only_route_result(
        self,
        *,
        node_ids: torch.Tensor,
        seed_candidate_indices: torch.Tensor,
        like: torch.Tensor,
    ) -> Dict[str, Any]:
        node_ids_list = [int(node_id) for node_id in node_ids.tolist()]
        reader_indices = self._seed_reader_node_indices(
            seed_candidate_indices=seed_candidate_indices,
            num_nodes=len(node_ids_list),
        )
        zero = like.new_zeros(())
        selected_ids = [node_ids_list[idx] for idx in reader_indices]
        return {
            "selected_node_indices": list(reader_indices),
            "selected_node_ids": list(selected_ids),
            "start_node_ids": list(selected_ids),
            "selected_edges": [],
            "hops_used": 0,
            "route_log_prob": zero,
            "route_entropy": zero,
            "frontier_candidate_count": 0,
            "seed_tail_candidate_count": 0,
            "frontier_selected_count": 0,
            "seed_tail_selected_count": 0,
            "frontier_path_selected_count": 0,
            "fixed_seed_count": len(selected_ids),
            "mode": "seed_only",
        }

    def _build_reader_inputs(
        self,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        evidence_input_ids: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        reader_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        pieces: List[int] = []

        def extend(ids: torch.Tensor, mask: torch.Tensor) -> None:
            if ids.numel() <= 0:
                return
            valid = ids[mask.bool()]
            pieces.extend(int(token_id) for token_id in valid.tolist())

        extend(prefix_input_ids[0], prefix_attention_mask[0])
        for node_idx in reader_indices:
            extend(evidence_input_ids[node_idx], evidence_attention_mask[node_idx])
        extend(query_input_ids[0], query_attention_mask[0])

        target_tokens: List[int] = []
        if target_input_ids.numel() > 0:
            target_tokens = [int(token_id) for token_id in target_input_ids[0][target_attention_mask[0].bool()].tolist()]
            pieces.extend(target_tokens)

        if self.max_len > 0 and len(pieces) > self.max_len:
            raise ValueError(
                f"Routed reader prompt requires {len(pieces)} tokens, exceeding max_len={self.max_len}. "
                "Reduce compact memory size, beam_width, max_hops, or max_evidence_nodes."
            )

        input_ids = torch.tensor([pieces], dtype=torch.long, device=prefix_input_ids.device)
        attention_mask = torch.ones_like(input_ids)
        labels = None
        logits_to_keep = 0
        if target_tokens:
            # Match the raw baseline memory optimization: only keep the logits
            # needed to score the target suffix, not the full routed prefix.
            labels = torch.tensor(
                [[-100] + target_tokens],
                dtype=torch.long,
                device=prefix_input_ids.device,
            )
            logits_to_keep = len(target_tokens) + 1
        return input_ids, attention_mask, labels, logits_to_keep

    def _compute_answer_router_loss(
        self,
        *,
        answer_loss: torch.Tensor,
        route_result: Dict[str, Any],
        reference_answer_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        device = answer_loss.device
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        metrics: Dict[str, float] = {
            "answer_router_loss": 0.0,
            "answer_router_policy_loss": 0.0,
            "answer_router_entropy": 0.0,
            "answer_router_advantage": 0.0,
            "answer_router_baseline": 0.0,
            "answer_router_reference_loss": 0.0,
        }
        if not self.training or float(self.answer_router_loss_weight) <= 0.0:
            return total_loss, metrics

        route_log_prob = route_result.get("route_log_prob")
        route_entropy = route_result.get("route_entropy")
        if not isinstance(route_log_prob, torch.Tensor) or not isinstance(route_entropy, torch.Tensor):
            raise ValueError("Answer-router policy loss requires a stochastic route_log_prob and route_entropy.")

        answer_loss_value = float(answer_loss.detach().item())
        if reference_answer_loss is not None:
            baseline_value = float(reference_answer_loss.detach().item())
            advantage = answer_loss.detach() - reference_answer_loss.detach()
        else:
            if self._answer_loss_baseline is None:
                self._answer_loss_baseline = answer_loss_value
            baseline_value = float(self._answer_loss_baseline)
            advantage = answer_loss.detach() - answer_loss.new_tensor(baseline_value)
        policy_loss = advantage * route_log_prob
        total_loss = total_loss + float(self.answer_router_loss_weight) * policy_loss
        if float(self.answer_router_entropy_weight) > 0.0:
            total_loss = total_loss - float(self.answer_router_entropy_weight) * route_entropy

        if reference_answer_loss is None:
            momentum = float(self.answer_router_baseline_momentum)
            self._answer_loss_baseline = momentum * baseline_value + (1.0 - momentum) * answer_loss_value
        metrics.update(
            {
                "answer_router_loss": float(total_loss.detach().item()),
                "answer_router_policy_loss": float(policy_loss.detach().item()),
                "answer_router_entropy": float(route_entropy.detach().item()),
                "answer_router_advantage": float(advantage.detach().item()),
                "answer_router_baseline": baseline_value,
                "answer_router_reference_loss": baseline_value if reference_answer_loss is not None else 0.0,
            }
        )
        return total_loss, metrics

    def forward(
        self,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        node_ids: torch.Tensor,
        seed_candidate_indices: torch.Tensor,
        node_sum_token_ids: torch.Tensor,
        node_sum_token_mask: torch.Tensor,
        evidence_input_ids: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        node_retrieval_scores: torch.Tensor,
        edge_src_index: torch.Tensor,
        edge_dst_index: torch.Tensor,
        edge_link_token_ids: torch.Tensor,
        edge_link_token_mask: torch.Tensor,
        edge_types: List[List[str]],
        edge_link_tokens: List[List[List[str]]],
        efficiency_stats: List[Dict[str, float]],
        **_: Any,
    ):
        _, _, _, start_logits, edge_logits = self._encode_router(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            node_sum_token_ids=node_sum_token_ids,
            node_sum_token_mask=node_sum_token_mask,
            edge_src_index=edge_src_index,
            edge_dst_index=edge_dst_index,
            edge_link_token_ids=edge_link_token_ids,
            edge_link_token_mask=edge_link_token_mask,
        )
        if self.use_seed_evidence_only:
            route_result = self._seed_only_route_result(
                node_ids=node_ids[0],
                seed_candidate_indices=seed_candidate_indices[0],
                like=start_logits,
            )
            reader_indices = list(route_result["selected_node_indices"])
        else:
            route_result = self._frontier_route(
                node_ids=node_ids[0],
                seed_candidate_indices=seed_candidate_indices[0],
                node_retrieval_scores=node_retrieval_scores[0],
                edge_src_index=edge_src_index[0],
                edge_dst_index=edge_dst_index[0],
                edge_types=edge_types[0],
                edge_link_tokens=edge_link_tokens[0],
                start_logits=start_logits,
                edge_logits=edge_logits,
                sample=self.training,
            )
            reader_indices = self._reader_node_indices(route_result=route_result)
        answer_input_ids, answer_attention_mask, answer_labels, answer_logits_to_keep = self._build_reader_inputs(
            prefix_input_ids=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            evidence_input_ids=evidence_input_ids[0],
            evidence_attention_mask=evidence_attention_mask[0],
            reader_indices=reader_indices,
        )
        outputs = self.base_model(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
            labels=answer_labels,
            logits_to_keep=answer_logits_to_keep,
            return_dict=True,
        )

        router_metrics: Dict[str, float] = {}
        total_loss = start_logits.new_zeros(())
        if outputs.loss is not None:
            total_loss = total_loss + outputs.loss
            router_metrics["answer_loss"] = float(outputs.loss.detach().item())
            if self.training and not self.use_seed_evidence_only and float(self.answer_router_loss_weight) > 0.0:
                reference_answer_loss = None
                baseline_count = len(route_result["selected_node_ids"])
                if int(self.frontier_baseline_seed_count) > 0:
                    baseline_count = min(baseline_count, int(self.frontier_baseline_seed_count))
                if baseline_count <= 0:
                    baseline_count = int(self.max_evidence_nodes)
                baseline_indices = self._valid_seed_indices(
                    seed_candidate_indices=seed_candidate_indices[0],
                    num_nodes=int(node_ids.shape[1]),
                )[:baseline_count]
                baseline_indices = baseline_indices[: self.max_evidence_nodes]
                baseline_indices.sort()
                (
                    baseline_input_ids,
                    baseline_attention_mask,
                    baseline_labels,
                    baseline_logits_to_keep,
                ) = self._build_reader_inputs(
                    prefix_input_ids=prefix_input_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    target_input_ids=target_input_ids,
                    target_attention_mask=target_attention_mask,
                    evidence_input_ids=evidence_input_ids[0],
                    evidence_attention_mask=evidence_attention_mask[0],
                    reader_indices=baseline_indices,
                )
                with torch.no_grad():
                    baseline_outputs = self.base_model(
                        input_ids=baseline_input_ids,
                        attention_mask=baseline_attention_mask,
                        labels=baseline_labels,
                        logits_to_keep=baseline_logits_to_keep,
                        return_dict=True,
                    )
                if baseline_outputs.loss is None:
                    raise ValueError("Frontier answer-router baseline requires answer labels and answer_loss.")
                reference_answer_loss = baseline_outputs.loss.detach()
                answer_router_loss, answer_router_metrics = self._compute_answer_router_loss(
                    answer_loss=outputs.loss,
                    route_result=route_result,
                    reference_answer_loss=reference_answer_loss,
                )
                total_loss = total_loss + answer_router_loss
                router_metrics.update(answer_router_metrics)
            else:
                router_metrics.update(
                    {
                        "answer_router_loss": 0.0,
                        "answer_router_policy_loss": 0.0,
                        "answer_router_entropy": 0.0,
                        "answer_router_advantage": 0.0,
                        "answer_router_baseline": 0.0,
                        "answer_router_reference_loss": 0.0,
                    }
                )
        else:
            router_metrics["answer_loss"] = 0.0
            if self.training and float(self.answer_router_loss_weight) > 0.0:
                raise ValueError("Answer-router policy loss requires answer labels and answer_loss.")

        self._last_route_result = {
            key: value
            for key, value in route_result.items()
            if key not in {"route_log_prob", "route_entropy"}
        }
        self._last_evidence_node_ids = [int(node_ids[0, idx].item()) for idx in reader_indices]
        self._last_generation_prompt_len = int(answer_input_ids.shape[1])
        self._last_efficiency_stats = {
            **dict(efficiency_stats[0] or {}),
            "sequence_len": float(answer_input_ids.shape[1]),
            "evidence_nodes": float(len(reader_indices)),
            "selected_nodes": float(len(route_result["selected_node_ids"])),
            "route_hops": float(route_result["hops_used"]),
            "frontier_candidates": float(route_result.get("frontier_candidate_count", 0)),
            "frontier_selected": float(route_result.get("frontier_selected_count", 0)),
            "frontier_path_selected": float(route_result.get("frontier_path_selected_count", 0)),
            "seed_tail_candidates": float(route_result.get("seed_tail_candidate_count", 0)),
            "seed_tail_selected": float(route_result.get("seed_tail_selected_count", 0)),
            "fixed_seed_nodes": float(route_result.get("fixed_seed_count", 0)),
        }
        self._last_training_metrics = {
            "total_loss": float(total_loss.detach().item()),
            "route_log_prob": float(route_result["route_log_prob"].detach().item())
            if isinstance(route_result.get("route_log_prob"), torch.Tensor)
            else 0.0,
            "route_entropy": float(route_result["route_entropy"].detach().item())
            if isinstance(route_result.get("route_entropy"), torch.Tensor)
            else 0.0,
            **router_metrics,
            **{
                f"eff_{key}": float(value)
                for key, value in self._last_efficiency_stats.items()
            },
        }
        outputs.loss = total_loss
        for key, value in router_metrics.items():
            setattr(outputs, key, value)
        return outputs

    def generate_routed(
        self,
        *,
        record_id: str | None = None,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        node_ids: torch.Tensor,
        seed_candidate_indices: torch.Tensor,
        node_sum_token_ids: torch.Tensor,
        node_sum_token_mask: torch.Tensor,
        evidence_input_ids: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        node_retrieval_scores: torch.Tensor,
        edge_src_index: torch.Tensor,
        edge_dst_index: torch.Tensor,
        edge_link_token_ids: torch.Tensor,
        edge_link_token_mask: torch.Tensor,
        edge_types: List[List[str]],
        edge_link_tokens: List[List[List[str]]],
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        efficiency_stats: List[Dict[str, float]],
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        del record_id
        _, _, _, start_logits, edge_logits = self._encode_router(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            node_sum_token_ids=node_sum_token_ids,
            node_sum_token_mask=node_sum_token_mask,
            edge_src_index=edge_src_index,
            edge_dst_index=edge_dst_index,
            edge_link_token_ids=edge_link_token_ids,
            edge_link_token_mask=edge_link_token_mask,
        )
        if self.use_seed_evidence_only:
            route_result = self._seed_only_route_result(
                node_ids=node_ids[0],
                seed_candidate_indices=seed_candidate_indices[0],
                like=start_logits,
            )
            reader_indices = list(route_result["selected_node_indices"])
        else:
            route_result = self._frontier_route(
                node_ids=node_ids[0],
                seed_candidate_indices=seed_candidate_indices[0],
                node_retrieval_scores=node_retrieval_scores[0],
                edge_src_index=edge_src_index[0],
                edge_dst_index=edge_dst_index[0],
                edge_types=edge_types[0],
                edge_link_tokens=edge_link_tokens[0],
                start_logits=start_logits,
                edge_logits=edge_logits,
                sample=False,
            )
            reader_indices = list(route_result["selected_node_indices"][: self.max_evidence_nodes])
            reader_indices.sort()
        answer_input_ids, answer_attention_mask, _labels, _logits_to_keep = self._build_reader_inputs(
            prefix_input_ids=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            evidence_input_ids=evidence_input_ids[0],
            evidence_attention_mask=evidence_attention_mask[0],
            reader_indices=reader_indices,
        )
        self._last_route_result = dict(route_result)
        self._last_evidence_node_ids = [int(node_ids[0, idx].item()) for idx in reader_indices]
        self._last_generation_prompt_len = int(answer_input_ids.shape[1])
        self._last_efficiency_stats = {
            **dict(efficiency_stats[0] or {}),
            "sequence_len": float(answer_input_ids.shape[1]),
            "evidence_nodes": float(len(reader_indices)),
            "selected_nodes": float(len(route_result["selected_node_ids"])),
            "route_hops": float(route_result["hops_used"]),
            "frontier_candidates": float(route_result.get("frontier_candidate_count", 0)),
            "frontier_selected": float(route_result.get("frontier_selected_count", 0)),
            "frontier_path_selected": float(route_result.get("frontier_path_selected_count", 0)),
            "seed_tail_candidates": float(route_result.get("seed_tail_candidate_count", 0)),
            "seed_tail_selected": float(route_result.get("seed_tail_selected_count", 0)),
            "fixed_seed_nodes": float(route_result.get("fixed_seed_count", 0)),
        }
        return self.base_model.generate(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
            **generate_kwargs,
        )


def summarize_dataset_efficiency(model: RoutedReaderModel, rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for row in rows:
        count += 1
        for key, value in (row or {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if count <= 0:
        return {}
    return {f"eff_{key}_avg": value / count for key, value in totals.items()}


def evaluate_generation(
    model: RoutedReaderModel,
    tokenizer: Any,
    dataset: RoutedReaderDataset,
    *,
    device: str,
    out_dir: str,
    max_new_tokens: int,
) -> Dict[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    predictions_path = Path(out_dir) / "predictions.jsonl"
    metric_rows: List[Dict[str, float]] = []
    efficiency_rows: List[Dict[str, float]] = []
    seen_questions: set[tuple[str, str]] = set()
    collator = RoutedReaderCollator(tokenizer)
    model.eval()
    target_end_token_id = strict_special_token_id(tokenizer, "</target>")
    eos_token_ids: List[int] = [target_end_token_id]
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) != target_end_token_id:
        eos_token_ids.insert(0, int(tokenizer.eos_token_id))

    with predictions_path.open("w", encoding="utf-8") as f_out:
        progress = tqdm(enumerate(dataset.records), total=len(dataset.records), desc="lm eval", unit="record")
        for idx, record in progress:
            question_key = qasper_question_key(record)
            if question_key in seen_questions:
                continue
            seen_questions.add(question_key)
            batch = collator([dataset.generation_item(idx)])
            generation_inputs: Dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    generation_inputs[key] = value.to(device)
                else:
                    generation_inputs[key] = value
            with torch.no_grad():
                outputs = model.generate_routed(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0],
                )
            prompt_len = int(model._last_generation_prompt_len)
            generated = outputs[0, prompt_len:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            answer_scores = best_answer_scores(
                prediction,
                [str(item.get("normalized_answer") or "") for item in (record.get("answer_refs") or [])],
            )
            evidence_scores = best_evidence_scores(
                model._last_evidence_node_ids,
                record.get("gold_evidence_sets") or [],
            )
            row = {
                "id": record["id"],
                "paper_id": question_key[0],
                "question_id": question_key[1],
                "prediction": prediction,
                "selected_node_ids": list(model._last_route_result.get("selected_node_ids") or []),
                "route_start_node_ids": list(model._last_route_result.get("start_node_ids") or []),
                "route_edges": list(model._last_route_result.get("selected_edges") or []),
                "predicted_evidence_node_ids": list(model._last_evidence_node_ids),
                "reference_answers": [str(item.get("normalized_answer") or "") for item in record.get("answer_refs", [])],
                **answer_scores,
                **evidence_scores,
            }
            metric_rows.append(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key.endswith("_f1") or key.endswith("_em")
                }
            )
            efficiency_rows.append(dict(model._last_efficiency_stats))
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.set_postfix(wrote=len(metric_rows))

    metrics = aggregate_prediction_metrics(metric_rows)
    metrics.update(summarize_dataset_efficiency(model, efficiency_rows))
    save_json(str(Path(out_dir) / "metrics.json"), metrics)
    return metrics


class RoutedReaderTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        trainer_log_path: str,
        train_step_metrics_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainer_log_path = Path(trainer_log_path)
        self.train_step_metrics_path = Path(train_step_metrics_path)
        self._loss_call_count = 0
        self.trainer_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer_log_path.write_text("", encoding="utf-8")
        self.train_step_metrics_path.write_text("", encoding="utf-8")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if loss is None:
            raise ValueError("Model returned no loss during training.")
        self._loss_call_count += 1
        metrics = dict(getattr(model, "_last_training_metrics", {}) or {})
        row = {
            "event": "train_step",
            "loss_call": int(self._loss_call_count),
            "global_step": int(self.state.global_step),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            **{key: json_safe(value) for key, value in metrics.items()},
        }
        append_jsonl(self.train_step_metrics_path, row)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        enriched_logs = dict(logs)
        metrics = dict(getattr(self.model, "_last_training_metrics", {}) or {})
        for key, value in metrics.items():
            enriched_logs.setdefault(key, json_safe(value))
        super().log(enriched_logs, start_time=start_time)
        row = {
            "event": "trainer_log",
            "global_step": int(self.state.global_step),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            **{key: json_safe(value) for key, value in enriched_logs.items()},
        }
        append_jsonl(self.trainer_log_path, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--max-evidence-nodes", type=int, default=DEFAULT_MAX_EVIDENCE_NODES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-head-hidden", type=int, default=DEFAULT_GRAPH_HEAD_HIDDEN)
    parser.add_argument("--graph-head-dropout", type=float, default=DEFAULT_GRAPH_HEAD_DROPOUT)
    parser.add_argument("--route-seed-limit", type=int, default=16)
    parser.add_argument("--answer-router-loss-weight", type=float, default=0.05)
    parser.add_argument("--answer-router-baseline-momentum", type=float, default=DEFAULT_ANSWER_ROUTER_BASELINE_MOMENTUM)
    parser.add_argument("--answer-router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--router-start-prior-weight", type=float, default=0.0)
    parser.add_argument("--router-edge-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--fixed-seed-evidence-count",
        type=int,
        default=CURRENT_PIPELINE_FIXED_SEED_EVIDENCE_COUNT,
    )
    parser.add_argument(
        "--frontier-select-count",
        type=int,
        default=CURRENT_PIPELINE_FRONTIER_SELECT_COUNT,
    )
    parser.add_argument("--frontier-hop-count", type=int, default=2)
    parser.add_argument("--frontier-baseline-seed-count", type=int, default=8)
    parser.add_argument("--frontier-max-per-seed", type=int, default=0)
    parser.add_argument("--query-conditioned-frontier-reranker", action="store_true")
    parser.add_argument(
        "--disable-query-conditioned-frontier-reranker",
        dest="query_conditioned_frontier_reranker",
        action="store_false",
    )
    parser.add_argument("--frontier-query-reranker-weight", type=float, default=2.0)
    parser.add_argument(
        "--disable-reader-summary-header",
        action="store_true",
        help="Do not prepend selected node summary phrases before each routed evidence paragraph.",
    )
    parser.add_argument(
        "--use-seed-evidence-only",
        action="store_true",
        help="Use the ranked seed candidates as reader evidence instead of routed selected nodes.",
    )
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.set_defaults(query_conditioned_frontier_reranker=True)
    args = parser.parse_args()

    if int(args.batch) != 1:
        raise ValueError("The final routed reader requires --batch 1.")
    if int(args.top_k) <= 0:
        raise ValueError(f"--top-k must be positive, got {args.top_k}.")
    if int(args.max_hops) <= 0:
        raise ValueError(f"--max-hops must be positive, got {args.max_hops}.")
    if int(args.beam_width) <= 0:
        raise ValueError(f"--beam-width must be positive, got {args.beam_width}.")
    if int(args.max_evidence_nodes) <= 0:
        raise ValueError(f"--max-evidence-nodes must be positive, got {args.max_evidence_nodes}.")
    if args.top_k < int(args.beam_width):
        raise ValueError("--top-k must be at least beam-width so every routed start can survive.")
    if int(args.graph_head_hidden) % 4 != 0:
        raise ValueError("--graph-head-hidden must be divisible by 4 for the router transformer encoder.")
    if int(args.route_seed_limit) < 0:
        raise ValueError("--route-seed-limit must be non-negative.")
    if args.use_seed_evidence_only:
        if float(args.answer_router_loss_weight) < 0.0:
            raise ValueError("--answer-router-loss-weight must be non-negative.")
    elif float(args.answer_router_loss_weight) <= 0.0:
        raise ValueError("--answer-router-loss-weight must be positive.")
    if not 0.0 <= float(args.answer_router_baseline_momentum) < 1.0:
        raise ValueError("--answer-router-baseline-momentum must be in [0, 1).")
    if float(args.answer_router_entropy_weight) < 0.0:
        raise ValueError("--answer-router-entropy-weight must be non-negative.")
    if float(args.router_start_prior_weight) < 0.0:
        raise ValueError("--router-start-prior-weight must be non-negative.")
    if float(args.router_edge_prior_weight) < 0.0:
        raise ValueError("--router-edge-prior-weight must be non-negative.")
    if int(args.fixed_seed_evidence_count) < 0:
        raise ValueError("--fixed-seed-evidence-count must be non-negative.")
    if int(args.frontier_select_count) < 0:
        raise ValueError("--frontier-select-count must be non-negative.")
    if int(args.frontier_hop_count) <= 0:
        raise ValueError("--frontier-hop-count must be positive.")
    if int(args.frontier_baseline_seed_count) < 0:
        raise ValueError("--frontier-baseline-seed-count must be non-negative.")
    if int(args.frontier_max_per_seed) < 0:
        raise ValueError("--frontier-max-per-seed must be non-negative.")
    if float(args.frontier_query_reranker_weight) < 0.0:
        raise ValueError("--frontier-query-reranker-weight must be non-negative.")
    total_frontier_nodes = int(args.fixed_seed_evidence_count) + int(args.frontier_select_count)
    if total_frontier_nodes > int(args.max_evidence_nodes):
        raise ValueError("--fixed-seed-evidence-count + --frontier-select-count must not exceed --max-evidence-nodes.")
    if total_frontier_nodes > int(args.top_k):
        raise ValueError("--fixed-seed-evidence-count + --frontier-select-count must not exceed --top-k.")
    if int(args.route_seed_limit) > 0 and int(args.route_seed_limit) < total_frontier_nodes:
        raise ValueError(
            "--route-seed-limit must be 0 or at least --fixed-seed-evidence-count + --frontier-select-count."
        )
    if int(args.frontier_baseline_seed_count) > 0 and int(args.frontier_baseline_seed_count) < total_frontier_nodes:
        raise ValueError(
            "--frontier-baseline-seed-count must be 0 or at least "
            "--fixed-seed-evidence-count + --frontier-select-count."
        )
    if int(args.frontier_max_per_seed) > 0 and int(args.frontier_select_count) > (
        int(args.fixed_seed_evidence_count) * int(args.frontier_max_per_seed)
    ):
        raise ValueError("--frontier-select-count exceeds --fixed-seed-evidence-count * --frontier-max-per-seed.")

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

    memory_vocab = build_memory_vocab(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False
    model_context_len = effective_model_max_length(tokenizer, model_config=base_model.config)
    if args.max_len > 0 and model_context_len > 0 and args.max_len > model_context_len:
        raise ValueError(
            f"--max-len={args.max_len} exceeds model context window {model_context_len}. No truncation is allowed."
        )
    effective_max_len = int(args.max_len) if int(args.max_len) > 0 else int(model_context_len)
    if effective_max_len <= 0:
        raise ValueError("Could not determine a valid max_len. Pass --max-len explicitly.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_json(
        str(Path(args.out) / "training_config.json"),
        {
            "train_qasper_routed_reader_args": json_safe(vars(args)),
            "argv": sys.argv,
            "resolved": {
                "model_path": str(model_path),
                "model_context_len": int(model_context_len),
                "effective_max_len": int(effective_max_len),
            },
        },
    )

    model = RoutedReaderModel(
        base_model=base_model,
        memory_vocab=memory_vocab,
        max_len=effective_max_len,
        top_k=args.top_k,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        max_evidence_nodes=args.max_evidence_nodes,
        hidden_dim=args.graph_head_hidden,
        dropout=args.graph_head_dropout,
        answer_router_loss_weight=args.answer_router_loss_weight,
        answer_router_baseline_momentum=args.answer_router_baseline_momentum,
        answer_router_entropy_weight=args.answer_router_entropy_weight,
        router_start_prior_weight=args.router_start_prior_weight,
        router_edge_prior_weight=args.router_edge_prior_weight,
        fixed_seed_evidence_count=args.fixed_seed_evidence_count,
        frontier_select_count=args.frontier_select_count,
        frontier_hop_count=args.frontier_hop_count,
        frontier_baseline_seed_count=args.frontier_baseline_seed_count,
        frontier_max_per_seed=args.frontier_max_per_seed,
        query_conditioned_frontier_reranker=args.query_conditioned_frontier_reranker,
        frontier_query_reranker_weight=args.frontier_query_reranker_weight,
        use_seed_evidence_only=args.use_seed_evidence_only,
    )

    train_dataset = RoutedReaderDataset(
        args.data,
        tokenizer,
        memory_vocab,
        max_len=effective_max_len,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        top_k=args.top_k,
        max_evidence_nodes=args.max_evidence_nodes,
        max_records=args.max_train,
        include_summary_header=False,
        route_seed_limit=args.route_seed_limit,
    )
    eval_dataset = None
    if args.eval_data:
        eval_dataset = RoutedReaderDataset(
            args.eval_data,
            tokenizer,
            memory_vocab,
            max_len=effective_max_len,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            top_k=args.top_k,
            max_evidence_nodes=args.max_evidence_nodes,
            max_records=args.max_eval,
            include_summary_header=False,
            route_seed_limit=args.route_seed_limit,
        )

    training_args = TrainingArguments(
        output_dir=str(Path(args.out) / "trainer_tmp"),
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
    )
    trainer = RoutedReaderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RoutedReaderCollator(tokenizer),
        trainer_log_path=str(Path(args.out) / "trainer_logs.jsonl"),
        train_step_metrics_path=str(Path(args.out) / "train_step_metrics.jsonl"),
    )
    train_result = trainer.train()
    save_json(str(Path(args.out) / "train_metrics.json"), json_safe(dict(train_result.metrics or {})))
    save_json(str(Path(args.out) / "trainer_log_history.json"), {"log_history": json_safe(trainer.state.log_history)})
    trainer.state.save_to_json(str(Path(args.out) / "trainer_state.json"))

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    save_json(str(Path(args.out) / "memory_vocab.json"), memory_vocab.to_dict())

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
