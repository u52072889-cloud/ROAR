#!/usr/bin/env python
"""Shared text and graph-memory helpers for phase1."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Sequence


SPECIAL_TOKENS = [
    "<query>",
    "</query>",
    "<summary>",
    "</summary>",
    "<evidence>",
    "</evidence>",
    "<target>",
    "</target>",
    "<prefix>",
    "</prefix>",
]

GRAPH_SCHEMA_VERSION = 3
TARGET_REGION_ID = -2
QUERY_REGION_ID = -3

KNOWN_MEMORY_PREFIXES = ("MEM_", "REL_")
NON_TOKEN_CHARS_RE = re.compile(r"[^A-Z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")
MODEL_MAX_LENGTH_SENTINEL = 1_000_000


def compact_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


def effective_model_max_length(tokenizer: Any, *, model_config: Any | None = None) -> int:
    candidates: List[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < MODEL_MAX_LENGTH_SENTINEL:
        candidates.append(int(tokenizer_limit))
    if model_config is not None:
        for field in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
            value = getattr(model_config, field, None)
            if isinstance(value, int) and value > 0:
                candidates.append(int(value))
    return min(candidates) if candidates else 0


def strict_encode(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = False,
    max_length: int = 0,
    context: str = "text",
    ) -> List[int]:
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if max_length and len(encoded) > max_length:
        raise ValueError(
            f"{context} token length {len(encoded)} exceeds configured limit {max_length}. "
            "No truncation is allowed in this pipeline."
        )
    return encoded


def strict_special_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer is missing required special token {token!r}.")
    if tokenizer.convert_ids_to_tokens(int(token_id)) != token:
        raise ValueError(
            f"Tokenizer mapped required special token {token!r} to id {token_id}, "
            f"but that id decodes to {tokenizer.convert_ids_to_tokens(int(token_id))!r}."
        )
    return int(token_id)


def resolve_local_hf_model_path(model_name_or_path: str, *, cache_dir: str = "") -> str:
    value = str(model_name_or_path or "").strip()
    if not value:
        raise ValueError("Empty model name/path.")

    direct = Path(value)
    if direct.exists():
        return str(direct)

    root = Path(cache_dir).expanduser() if cache_dir else None
    if root is not None and root.exists() and "/" in value:
        org, model = value.split("/", 1)
        repo_dir = root / f"models--{org}--{model}"
        snapshots_dir = repo_dir / "snapshots"
        refs_dir = repo_dir / "refs"
        if snapshots_dir.exists():
            ref_candidates: List[str] = []
            for ref_name in ("main", "master"):
                ref_file = refs_dir / ref_name
                if ref_file.exists():
                    ref_candidates.append(ref_file.read_text(encoding="utf-8").strip())
            for ref in ref_candidates:
                snapshot = snapshots_dir / ref
                if snapshot.exists():
                    return str(snapshot)
            snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if len(snapshots) == 1:
                return str(snapshots[0])
            if snapshots:
                return str(snapshots[-1])

    return value


def set_offline_hf_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def normalize_memory_token(token: str, *, default_prefix: str = "MEM_") -> str:
    token = NON_TOKEN_CHARS_RE.sub("_", str(token or "").upper()).strip("_")
    if not token:
        return f"{default_prefix}EMPTY"
    if token.startswith(KNOWN_MEMORY_PREFIXES):
        return token
    return f"{default_prefix}{token}"


def _expand_multi_mem_token(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    starts = [match.start() for match in re.finditer(r"MEM_", text.upper())]
    if len(starts) <= 1:
        return [text]
    expanded: List[str] = []
    starts.append(len(text))
    for index in range(len(starts) - 1):
        piece = text[starts[index] : starts[index + 1]].strip(" \t\r\n,;|/")
        if piece:
            expanded.append(piece)
    return expanded or [text]


def normalize_node_tokens(
    tokens: Sequence[str],
    *,
    fallback: Sequence[str] | None = None,
    max_tokens: int = 12,
) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for raw in list(tokens or []) + list(fallback or []):
        for candidate in _expand_multi_mem_token(raw):
            token = normalize_memory_token(candidate, default_prefix="MEM_")
            if not token.startswith("MEM_"):
                token = f"MEM_{token}"
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
            if max_tokens and len(ordered) >= max_tokens:
                break
        if max_tokens and len(ordered) >= max_tokens:
            break
    if not ordered:
        ordered.append("MEM_EMPTY")
    return ordered
