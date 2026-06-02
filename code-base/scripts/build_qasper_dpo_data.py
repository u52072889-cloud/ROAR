#!/usr/bin/env python
"""Build simple-scored DPO pairs from phase1 train artifacts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from data_utils import (  # noqa: E402
    compact_whitespace,
    effective_model_max_length,
    resolve_local_hf_model_path,
    set_offline_hf_env,
)


DEFAULT_TRAIN_ARTIFACT = str(REPO_ROOT / "artifacts" / "qasper_train_graph.jsonl")
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_REWARD_EMBEDDER = "BAAI/bge-large-en-v1.5"
LEAKAGE_PATTERNS = (
    (re.compile(r"(?im)^\s*Section title\s*:", re.IGNORECASE), 0.5),
    (re.compile(r"(?im)^\s*Paragraph node\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"(?im)^\s*Evidence\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"(?im)^\s*Summary\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"(?im)^\s*Phrase\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"(?im)^\s*Summary\s+\d+\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"(?im)^\s*Rules\s*:", re.IGNORECASE), 1.0),
    (re.compile(r"\bWrite up to five concise retrieval phrases for one paragraph node\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bWrite 1 to 5 concise retrieval phrases for the paragraph node\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bEach phrase must use at most 5 words\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bWrite one phrase per line\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bOutput one phrase per line\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bPrefer exact key terms from the paragraph\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bKeep important names, datasets, methods, results, numbers, and comparisons\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bIf the paragraph is about an action or relation\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bKeep the subject and relation words together\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bDo not replace specific words with generic words\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bDo not write explanations, labels, bullets, or citations\b", re.IGNORECASE), 2.0),
    (re.compile(r"\bDo not use bullets, numbering, or separators\b", re.IGNORECASE), 2.0),
    (re.compile(r"(?im)^\s*(?:system|user|assistant)\s*:", re.IGNORECASE), 2.0),
    (re.compile(r"\bYou are Qwen\b", re.IGNORECASE), 3.0),
    (re.compile(r"\blabels or explanations\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bphrase per line\b", re.IGNORECASE), 0.5),
)
GROUNDING_SPLIT_RE = re.compile(r"\n+")
REDUNDANCY_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
MAX_SINGLE_PHRASE_WORDS = 5
MAX_OUTPUT_PHRASES = 5
LEAKAGE_REWARD_PENALTY_WEIGHT = 1.0
LONG_SINGLE_PHRASE_REWARD_PENALTY_WEIGHT = 1.0
LONG_OUTPUT_REWARD_PENALTY_WEIGHT = 1.0
DEFAULT_MIN_REWARD_MARGIN = 0.10
DEFAULT_MIN_CHOSEN_QUERY_SCORE = 0.20
DEFAULT_MIN_RAW_QUERY_SCORE = 0.30
DEFAULT_NUM_ROUNDS = 4

GENERATOR_SYSTEM_PROMPT = (
    "Write up to five concise retrieval phrases for one paragraph node. "
    "Output one phrase per line."
)


@dataclass
class EvidenceTask:
    record_id: str
    paper_id: str
    question_id: str
    annotation_id: str
    question: str
    answer: str
    node_id: int
    section_title: str
    evidence_text: str
    prompt: str
    scoring_questions: List[str]
    scoring_section_titles: List[str]
    raw_query_score: float = 0.0
    per_query_raw_query_scores: List[float] | None = None


def task_output_id(task: EvidenceTask) -> str:
    return f"{task.record_id}::node{task.node_id}"


def read_existing_output_ids(paths: Sequence[Path | None]) -> set[str]:
    existing_ids: set[str] = set()
    for path in paths:
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                row_id = str(payload.get("id") or "").strip()
                if not row_id:
                    raise ValueError(f"{path}:{line_number} is missing id; cannot resume.")
                existing_ids.add(row_id)
    return existing_ids


def format_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and str(getattr(tokenizer, "chat_template", "") or "").strip():
        messages = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if str(system_prompt or "").strip():
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def decode_generations(
    *,
    tokenizer: Any,
    outputs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[str]:
    prompt_width = int(attention_mask.shape[1])
    decoded: List[str] = []
    for row in range(int(outputs.shape[0])):
        gen_ids = outputs[row][prompt_width:]
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return decoded


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def require_texts_within_limit(
    *,
    tokenizer: Any,
    texts: Sequence[str],
    limit: int,
    context: str,
    reserve_tokens: int = 0,
) -> None:
    if limit <= 0:
        return
    for index, length in enumerate(token_lengths(tokenizer, texts), start=1):
        total = int(length) + int(reserve_tokens)
        if total > limit:
            raise ValueError(
                f"{context} item {index} requires {total} tokens including reserved generation budget, "
                f"which exceeds limit {limit}. No truncation is allowed."
            )


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


def load_hf_causal_lm(
    *,
    model_name_or_path: str,
    device: str,
    cache_dir: str,
    allow_download: bool,
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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    require_nonempty_tokenizer(tokenizer, context=f"Model '{model_name_or_path}'")

    model_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=not allow_download,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def load_hf_encoder(
    *,
    model_name_or_path: str,
    device: str,
    cache_dir: str,
    allow_download: bool,
) -> tuple[Any, Any]:
    model_path = resolve_local_hf_model_path(model_name_or_path, cache_dir=cache_dir)
    model_is_local = Path(model_path).exists()
    if not model_is_local and not allow_download:
        raise FileNotFoundError(
            f"Reward embedder '{model_name_or_path}' was not found in cache_dir='{cache_dir}' while offline mode is enabled. "
            "Pre-download the model into the cache or rerun with --allow-download."
        )
    source = model_load_source(model_name_or_path, str(model_path), allow_download=allow_download)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        use_fast=True,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
    )
    require_nonempty_tokenizer(tokenizer, context=f"Reward embedder '{model_name_or_path}'")
    model = AutoModel.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=not allow_download,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def embed_texts_for_reward(
    *,
    tokenizer: Any,
    model: Any,
    texts: Sequence[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
    summed = (outputs.last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(summed / denom, dim=-1)


def section_summary_embedding_scores(
    *,
    tokenizer: Any,
    model: Any,
    question: str,
    section_title: str,
    summaries: Sequence[str],
    device: str,
    max_length: int,
) -> List[float]:
    section_summary_texts = [
        compact_whitespace(f"{section_title}\n{summary}") if str(section_title or "").strip() else compact_whitespace(summary)
        for summary in summaries
    ]
    vectors = embed_texts_for_reward(
        tokenizer=tokenizer,
        model=model,
        texts=[compact_whitespace(question)] + section_summary_texts,
        device=device,
        max_length=int(max_length),
    )
    scores = torch.mv(vectors[1:], vectors[0])
    return [float(score.detach().cpu().item()) for score in scores]


def raw_node_query_embedding_scores(
    *,
    tokenizer: Any,
    model: Any,
    questions: Sequence[str],
    evidence_text: str,
    device: str,
    max_length: int,
) -> tuple[float, List[float]]:
    if not questions:
        raise ValueError("raw_node_query_embedding_scores requires at least one question.")
    scores: List[float] = []
    for question in questions:
        score = section_summary_embedding_scores(
            tokenizer=tokenizer,
            model=model,
            question=question,
            section_title="",
            summaries=[evidence_text],
            device=device,
            max_length=int(max_length),
        )[0]
        scores.append(float(score))
    return sum(scores) / float(len(scores)), scores


def annotate_raw_query_scores_for_tasks(
    *,
    tokenizer: Any,
    model: Any,
    tasks: Sequence[EvidenceTask],
    device: str,
    max_length: int,
) -> None:
    pair_questions: List[str] = []
    pair_node_texts: List[str] = []
    spans: List[tuple[int, int]] = []
    for task in tasks:
        start = len(pair_questions)
        for question in task.scoring_questions:
            pair_questions.append(question)
            pair_node_texts.append(task.evidence_text)
        end = len(pair_questions)
        if start == end:
            raise ValueError(f"Task {task_output_id(task)} has no scoring questions.")
        spans.append((start, end))
    if not pair_questions:
        return
    vectors = embed_texts_for_reward(
        tokenizer=tokenizer,
        model=model,
        texts=pair_questions + pair_node_texts,
        device=device,
        max_length=int(max_length),
    )
    count = len(pair_questions)
    scores = (vectors[:count] * vectors[count:]).sum(dim=-1).detach().cpu().tolist()
    for task, (start, end) in zip(tasks, spans):
        task_scores = [float(score) for score in scores[start:end]]
        task.per_query_raw_query_scores = task_scores
        task.raw_query_score = sum(task_scores) / float(len(task_scores))


def build_node_generation_prompt(*, section_title: str, node_text: str) -> str:
    title_text = compact_whitespace(section_title) or "Unknown"
    paragraph_text = compact_whitespace(node_text)
    if not paragraph_text:
        raise ValueError("Cannot build a generation prompt for an empty paragraph node.")
    return "\n".join(
        [
            "Write 1 to 5 concise retrieval phrases for the paragraph node.",
            "Each phrase must use at most 5 words.",
            "Write one phrase per line.",
            "Prefer exact key terms from the paragraph.",
            "When a phrase captures a relation or comparison, keep the related entities and relation/comparison words together.",
            "Do not replace specific words with generic words.",
            "Do not write explanations, labels, bullets, or citations.",
            "Do not use bullets, numbering, or separators.",
            "",
            "Section title:",
            title_text,
            "",
            "Paragraph node:",
            paragraph_text,
        ]
    )


def simple_normalize_for_scoring(raw_output: str) -> str:
    text = str(raw_output or "").strip()
    if not text:
        raise ValueError("Generation returned empty raw output after trimming.")
    return text


def split_output_phrases(raw_output: str) -> List[str]:
    phrases: List[str] = []
    for raw_part in GROUNDING_SPLIT_RE.split(str(raw_output or "")):
        phrase = compact_whitespace(raw_part)
        if not phrase:
            continue
        phrases.append(phrase)
    if phrases:
        return phrases
    text = simple_normalize_for_scoring(raw_output)
    return [compact_whitespace(text)]


def simple_normalize_for_grounding(text: str) -> str:
    return compact_whitespace(str(text or "")).lower()


def redundancy_penalty(raw_output: str) -> float:
    normalized_phrases: List[str] = []
    for phrase in split_output_phrases(raw_output):
        normalized = compact_whitespace(REDUNDANCY_NORMALIZE_RE.sub(" ", phrase.lower()))
        if normalized:
            normalized_phrases.append(normalized)
    if len(normalized_phrases) < 2:
        return 0.0
    duplicates = len(normalized_phrases) - len(set(normalized_phrases))
    if duplicates <= 0:
        return 0.0
    return float(duplicates) / float(len(normalized_phrases))


def long_single_phrase_penalty(raw_output: str) -> float:
    penalty = 0.0
    for phrase in split_output_phrases(raw_output):
        word_count = len(phrase.split())
        if word_count > int(MAX_SINGLE_PHRASE_WORDS):
            penalty += float(word_count - int(MAX_SINGLE_PHRASE_WORDS))
    return penalty


def long_output_penalty(raw_output: str, *, max_phrases: int) -> float:
    phrase_count = len(split_output_phrases(raw_output))
    if phrase_count <= int(max_phrases):
        return 0.0
    return float(phrase_count - int(max_phrases))


def leakage_penalty(raw_output: str) -> float:
    text = str(raw_output or "")
    penalty = 0.0
    for pattern, weight in LEAKAGE_PATTERNS:
        penalty += float(weight) * len(pattern.findall(text))
    return penalty


def score_candidates(
    *,
    reward_embedder_tokenizer: Any,
    reward_embedder_model: Any,
    questions: Sequence[str],
    section_titles: Sequence[str],
    evidence_text: str,
    raw_outputs: Sequence[str],
    device: str,
    reward_embedder_max_length: int,
    raw_node_embedding_score: float | None = None,
    per_query_raw_node_embedding_scores: Sequence[float] | None = None,
) -> Dict[str, Any]:
    if len(raw_outputs) < 2:
        raise ValueError(f"Expected at least 2 raw outputs, received {len(raw_outputs)}.")
    if not questions:
        raise ValueError("score_candidates requires at least one scoring question.")
    if len(questions) != len(section_titles):
        raise ValueError(f"Expected {len(questions)} section titles, received {len(section_titles)}.")
    phrase_groups = [split_output_phrases(output) for output in raw_outputs]
    scoring_texts = ["\n".join(phrases) for phrases in phrase_groups]
    per_query_embedding_scores: List[List[float]] = []
    score_sums = [0.0 for _ in scoring_texts]
    for question, section_title in zip(questions, section_titles):
        scores = section_summary_embedding_scores(
            tokenizer=reward_embedder_tokenizer,
            model=reward_embedder_model,
            question=question,
            section_title=section_title,
            summaries=scoring_texts,
            device=device,
            max_length=int(reward_embedder_max_length),
        )
        per_query_embedding_scores.append(scores)
        for index, score in enumerate(scores):
            score_sums[index] += float(score)
    if raw_node_embedding_score is None or per_query_raw_node_embedding_scores is None:
        raw_node_embedding_score, raw_node_embedding_scores = raw_node_query_embedding_scores(
            tokenizer=reward_embedder_tokenizer,
            model=reward_embedder_model,
            questions=questions,
            evidence_text=evidence_text,
            device=device,
            max_length=int(reward_embedder_max_length),
        )
    else:
        raw_node_embedding_scores = [float(score) for score in per_query_raw_node_embedding_scores]
        if len(raw_node_embedding_scores) != len(questions):
            raise ValueError(
                f"Expected {len(questions)} raw node scores, received {len(raw_node_embedding_scores)}."
            )
        raw_node_embedding_score = float(raw_node_embedding_score)
    embedding_query_scores = [score_sum / float(len(per_query_embedding_scores)) for score_sum in score_sums]
    query_relevance_scores = [
        float(score) - float(raw_node_embedding_score)
        for score in embedding_query_scores
    ]
    max_output_phrases = MAX_OUTPUT_PHRASES
    penalties = [float(leakage_penalty(output)) for output in raw_outputs]
    long_single_phrase_penalties = [float(long_single_phrase_penalty(output)) for output in raw_outputs]
    long_output_penalties = [
        float(long_output_penalty(output, max_phrases=max_output_phrases))
        for output in raw_outputs
    ]
    rewards = [
        float(query_relevance_scores[index])
        - LEAKAGE_REWARD_PENALTY_WEIGHT * float(penalties[index])
        - LONG_SINGLE_PHRASE_REWARD_PENALTY_WEIGHT * float(long_single_phrase_penalties[index])
        - LONG_OUTPUT_REWARD_PENALTY_WEIGHT * float(long_output_penalties[index])
        for index in range(len(raw_outputs))
    ]
    return {
        "scoring_texts": scoring_texts,
        "max_output_phrases": max_output_phrases,
        "section_summary_embedding_scores": embedding_query_scores,
        "per_query_section_summary_embedding_scores": per_query_embedding_scores,
        "raw_node_embedding_score": raw_node_embedding_score,
        "per_query_raw_node_embedding_scores": raw_node_embedding_scores,
        "summary_raw_embedding_delta_scores": query_relevance_scores,
        "query_relevance_scores": query_relevance_scores,
        "leakage_penalties": penalties,
        "long_single_phrase_penalties": long_single_phrase_penalties,
        "long_output_penalties": long_output_penalties,
        "rewards": rewards,
    }


def choose_candidate_pair(score_info: Dict[str, List[float]]) -> tuple[int, int]:
    rewards = list(score_info["rewards"])
    if len(rewards) < 2:
        raise ValueError(f"Expected at least 2 rewards, received {len(rewards)}.")
    query_relevance_scores = list(score_info["query_relevance_scores"])
    leakage_penalties = list(score_info["leakage_penalties"])
    ranking = sorted(
        range(len(rewards)),
        key=lambda index: (
            float(rewards[index]),
            float(query_relevance_scores[index]),
            -float(leakage_penalties[index]),
        ),
        reverse=True,
    )
    return int(ranking[0]), int(ranking[-1])


def generate_two_raw_outputs(
    *,
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[List[str]]:
    if not prompts:
        return []
    context_window = effective_model_max_length(tokenizer, model_config=model.config)
    require_texts_within_limit(
        tokenizer=tokenizer,
        texts=prompts,
        limit=context_window,
        context="DPO generation prompt",
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    repeated_attention_mask = inputs["attention_mask"].repeat_interleave(2, dim=0)
    decoded = decode_generations(
        tokenizer=tokenizer,
        outputs=outputs,
        attention_mask=repeated_attention_mask,
    )
    if len(decoded) != len(prompts) * 2:
        raise ValueError(
            f"Expected {len(prompts) * 2} generated outputs, received {len(decoded)}."
        )
    grouped: List[List[str]] = []
    for index in range(0, len(decoded), 2):
        grouped.append(decoded[index : index + 2])
    return grouped


def generate_candidate_rounds(
    *,
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_rounds: int,
) -> List[List[str]]:
    if int(num_rounds) <= 0:
        raise ValueError("--num-rounds must be positive.")
    collected: List[List[str]] = [[] for _ in prompts]
    for _ in range(int(num_rounds)):
        round_outputs = generate_two_raw_outputs(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if len(round_outputs) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} round outputs, received {len(round_outputs)}.")
        for index, outputs in enumerate(round_outputs):
            collected[index].extend(list(outputs))
    return collected


def iter_evidence_tasks(
    *,
    artifact_path: str,
    max_records: int,
) -> Iterable[EvidenceTask]:
    processed = 0
    tasks_by_prompt: Dict[str, EvidenceTask] = {}
    seen_prompt_contexts: Dict[str, set[tuple[str, str]]] = {}
    with open(artifact_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            processed += 1
            if max_records and processed > max_records:
                break
            question = compact_whitespace(record.get("query", {}).get("question") or "")
            if not question:
                raise ValueError(f"Record {record.get('id')} is missing query.question.")
            nodes = {
                int(node["node_id"]): dict(node)
                for node in (record.get("nodes") or [])
            }
            if not nodes:
                raise ValueError(f"Record {record.get('id')} is missing nodes.")
            gold_node_ids: List[int] = []
            seen = set()
            for raw_node_id in record.get("gold_evidence_node_ids") or []:
                node_id = int(raw_node_id)
                if node_id in seen:
                    continue
                seen.add(node_id)
                gold_node_ids.append(node_id)
            if not gold_node_ids:
                continue
            for node_id in gold_node_ids:
                if node_id not in nodes:
                    raise ValueError(
                        f"Record {record.get('id')} gold evidence node_id={node_id} is missing from nodes."
                    )
                node = nodes[node_id]
                if str(node.get("node_type") or "") != "paragraph":
                    raise ValueError(
                        f"Record {record.get('id')} gold evidence node_id={node_id} is not a paragraph node."
                    )
                node_text = compact_whitespace(str(node.get("text") or ""))
                if not node_text:
                    raise ValueError(
                        f"Record {record.get('id')} gold evidence node_id={node_id} has empty text."
                    )
                section_title = str(node.get("section_title") or "").strip()
                answer = str(record.get("target_answer", {}).get("normalized_answer") or "").strip()
                prompt = build_node_generation_prompt(section_title=section_title, node_text=node_text)
                context_key = (question, section_title)
                existing_task = tasks_by_prompt.get(prompt)
                if existing_task is not None:
                    if context_key not in seen_prompt_contexts[prompt]:
                        existing_task.scoring_questions.append(question)
                        existing_task.scoring_section_titles.append(section_title)
                        seen_prompt_contexts[prompt].add(context_key)
                    continue
                tasks_by_prompt[prompt] = EvidenceTask(
                    record_id=str(record.get("id") or ""),
                    paper_id=str(record.get("paper_id") or ""),
                    question_id=str(record.get("question_id") or ""),
                    annotation_id=str(record.get("annotation_id") or ""),
                    question=question,
                    answer=answer,
                    node_id=node_id,
                    section_title=section_title,
                    evidence_text=node_text,
                    prompt=prompt,
                    scoring_questions=[question],
                    scoring_section_titles=[section_title],
                )
                seen_prompt_contexts[prompt] = {context_key}
    yield from tasks_by_prompt.values()


def raw_query_skip_payload(task: EvidenceTask, *, min_raw_query_score: float) -> Dict[str, Any]:
    return {
        "id": task_output_id(task),
        "source_record_id": task.record_id,
        "paper_id": task.paper_id,
        "question_id": task.question_id,
        "annotation_id": task.annotation_id,
        "question": task.question,
        "answer": task.answer,
        "node_id": int(task.node_id),
        "sample_scope": "node",
        "section_title": task.section_title,
        "scored_questions": list(task.scoring_questions),
        "scored_section_titles": list(task.scoring_section_titles),
        "prompt_group_size": len(task.scoring_questions),
        "evidence_text": task.evidence_text,
        "raw_query_score": float(task.raw_query_score),
        "per_query_raw_query_scores": list(task.per_query_raw_query_scores or []),
        "min_raw_query_score_threshold": float(min_raw_query_score),
        "skip_reason": "raw_query_similarity",
    }


def filter_tasks_by_raw_query_score(
    *,
    tasks: Sequence[EvidenceTask],
    skipped_handle: Any | None,
    reward_embedder_tokenizer: Any,
    reward_embedder_model: Any,
    device: str,
    reward_embedder_max_length: int,
    min_raw_query_score: float,
) -> tuple[List[EvidenceTask], int]:
    annotate_raw_query_scores_for_tasks(
        tokenizer=reward_embedder_tokenizer,
        model=reward_embedder_model,
        tasks=tasks,
        device=device,
        max_length=int(reward_embedder_max_length),
    )
    kept_tasks: List[EvidenceTask] = []
    skipped_low_raw_similarity = 0
    for task in tasks:
        if task.raw_query_score < float(min_raw_query_score):
            if skipped_handle is not None:
                skipped_handle.write(
                    json.dumps(
                        raw_query_skip_payload(task, min_raw_query_score=float(min_raw_query_score)),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            skipped_low_raw_similarity += 1
            continue
        kept_tasks.append(task)
    return kept_tasks, skipped_low_raw_similarity


def write_batch(
    *,
    handle: Any,
    skipped_handle: Any | None,
    tasks: Sequence[EvidenceTask],
    prompts: Sequence[str],
    raw_outputs: Sequence[Sequence[str]],
    reward_embedder_tokenizer: Any,
    reward_embedder_model: Any,
    device: str,
    min_reward_margin: float,
    min_chosen_query_score: float,
    reward_embedder_max_length: int,
) -> tuple[int, int, int, int]:
    wrote = 0
    skipped_ties = 0
    skipped_margin = 0
    skipped_low_similarity = 0
    if len(prompts) != len(tasks):
        raise ValueError(f"Expected {len(tasks)} prompts, received {len(prompts)}.")
    for task, prompt_text, candidate_outputs in zip(tasks, prompts, raw_outputs):
        if len(candidate_outputs) < 2:
            raise ValueError(
                f"Task {task.record_id} node_id={task.node_id} did not receive at least 2 raw outputs."
            )
        score_info = score_candidates(
            reward_embedder_tokenizer=reward_embedder_tokenizer,
            reward_embedder_model=reward_embedder_model,
            questions=task.scoring_questions,
            section_titles=task.scoring_section_titles,
            evidence_text=task.evidence_text,
            raw_outputs=candidate_outputs,
            device=device,
            reward_embedder_max_length=int(reward_embedder_max_length),
            raw_node_embedding_score=(
                task.raw_query_score if task.per_query_raw_query_scores is not None else None
            ),
            per_query_raw_node_embedding_scores=task.per_query_raw_query_scores,
        )
        rewards = list(score_info["rewards"])
        if len(rewards) < 2:
            raise ValueError(
                f"Task {task.record_id} node_id={task.node_id} did not receive at least 2 rewards."
            )
        chosen_index, rejected_index = choose_candidate_pair(score_info)
        reward_margin = float(rewards[chosen_index]) - float(rewards[rejected_index])
        chosen_query_embedding_scores: List[float] = []
        for query_scores in score_info["per_query_section_summary_embedding_scores"]:
            if chosen_index >= len(query_scores):
                raise ValueError(
                    f"Task {task.record_id} node_id={task.node_id} chosen_index={chosen_index} "
                    f"is out of range for per-query scores."
                )
            chosen_query_embedding_scores.append(float(query_scores[chosen_index]))
        min_chosen_query_embedding_score = min(chosen_query_embedding_scores)
        payload = {
            "id": task_output_id(task),
            "source_record_id": task.record_id,
            "paper_id": task.paper_id,
            "question_id": task.question_id,
            "annotation_id": task.annotation_id,
            "question": task.question,
            "answer": task.answer,
            "node_id": int(task.node_id),
            "sample_scope": "node",
            "section_title": task.section_title,
            "scored_questions": list(task.scoring_questions),
            "scored_section_titles": list(task.scoring_section_titles),
            "prompt_group_size": len(task.scoring_questions),
            "evidence_text": task.evidence_text,
            "prompt": prompt_text,
            "raw_outputs": list(candidate_outputs),
            "scoring_texts": list(score_info["scoring_texts"]),
            "max_output_phrases": int(score_info["max_output_phrases"]),
            "section_summary_embedding_scores": list(score_info["section_summary_embedding_scores"]),
            "per_query_section_summary_embedding_scores": list(
                score_info["per_query_section_summary_embedding_scores"]
            ),
            "raw_node_embedding_score": float(score_info["raw_node_embedding_score"]),
            "per_query_raw_node_embedding_scores": list(
                score_info["per_query_raw_node_embedding_scores"]
            ),
            "raw_query_score": float(score_info["raw_node_embedding_score"]),
            "per_query_raw_query_scores": list(score_info["per_query_raw_node_embedding_scores"]),
            "summary_raw_embedding_delta_scores": list(score_info["summary_raw_embedding_delta_scores"]),
            "query_relevance_scores": list(score_info["query_relevance_scores"]),
            "chosen_per_query_section_summary_embedding_scores": chosen_query_embedding_scores,
            "min_chosen_query_embedding_score": float(min_chosen_query_embedding_score),
            "min_chosen_query_score_threshold": float(min_chosen_query_score),
            "leakage_reward_penalty_weight": float(LEAKAGE_REWARD_PENALTY_WEIGHT),
            "long_single_phrase_reward_penalty_weight": float(LONG_SINGLE_PHRASE_REWARD_PENALTY_WEIGHT),
            "long_output_reward_penalty_weight": float(LONG_OUTPUT_REWARD_PENALTY_WEIGHT),
            "leakage_penalties": list(score_info["leakage_penalties"]),
            "long_single_phrase_penalties": list(score_info["long_single_phrase_penalties"]),
            "long_output_penalties": list(score_info["long_output_penalties"]),
            "rewards": rewards,
            "chosen": candidate_outputs[chosen_index],
            "rejected": candidate_outputs[rejected_index],
            "chosen_reward": float(rewards[chosen_index]),
            "rejected_reward": float(rewards[rejected_index]),
            "reward_margin": float(reward_margin),
            "chosen_index": int(chosen_index),
            "rejected_index": int(rejected_index),
        }
        if chosen_index == rejected_index:
            if skipped_handle is not None:
                skipped_payload = dict(payload)
                skipped_payload["skip_reason"] = "tie"
                skipped_handle.write(json.dumps(skipped_payload, ensure_ascii=False) + "\n")
            skipped_ties += 1
            continue
        if min_chosen_query_embedding_score < float(min_chosen_query_score):
            if skipped_handle is not None:
                skipped_payload = dict(payload)
                skipped_payload["skip_reason"] = "chosen_similarity"
                skipped_handle.write(json.dumps(skipped_payload, ensure_ascii=False) + "\n")
            skipped_low_similarity += 1
            continue
        if reward_margin < float(min_reward_margin):
            if skipped_handle is not None:
                skipped_payload = dict(payload)
                skipped_payload["skip_reason"] = "margin"
                skipped_payload["min_reward_margin"] = float(min_reward_margin)
                skipped_handle.write(json.dumps(skipped_payload, ensure_ascii=False) + "\n")
            skipped_margin += 1
            continue
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        wrote += 1
    return wrote, skipped_ties, skipped_margin, skipped_low_similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-artifact", default=DEFAULT_TRAIN_ARTIFACT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--reward-embedder", default=DEFAULT_REWARD_EMBEDDER)
    parser.add_argument("--device", default="")
    parser.add_argument("--cache-dir", default="/tmp/phase1_hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skipped-out", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--min-reward-margin", type=float, default=DEFAULT_MIN_REWARD_MARGIN)
    parser.add_argument("--min-chosen-query-score", type=float, default=DEFAULT_MIN_CHOSEN_QUERY_SCORE)
    parser.add_argument("--min-raw-query-score", type=float, default=DEFAULT_MIN_RAW_QUERY_SCORE)
    parser.add_argument("--raw-score-batch-size", type=int, default=256)
    parser.add_argument("--reward-embedder-max-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if float(args.temperature) <= 0.0:
        raise ValueError("--temperature must be positive.")
    if float(args.top_p) <= 0.0 or float(args.top_p) > 1.0:
        raise ValueError("--top-p must be within (0, 1].")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative.")
    if int(args.num_rounds) <= 0:
        raise ValueError("--num-rounds must be positive.")
    if float(args.min_reward_margin) < 0.0:
        raise ValueError("--min-reward-margin must be non-negative.")
    if float(args.min_chosen_query_score) < 0.0:
        raise ValueError("--min-chosen-query-score must be non-negative.")
    if float(args.min_raw_query_score) < 0.0:
        raise ValueError("--min-raw-query-score must be non-negative.")
    if int(args.raw_score_batch_size) <= 0:
        raise ValueError("--raw-score-batch-size must be positive.")
    if int(args.reward_embedder_max_length) <= 0:
        raise ValueError("--reward-embedder-max-length must be positive.")

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        set_offline_hf_env()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator_tokenizer, generator_model = load_hf_causal_lm(
        model_name_or_path=args.generator_model,
        device=device,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )
    reward_embedder_tokenizer, reward_embedder_model = load_hf_encoder(
        model_name_or_path=args.reward_embedder,
        device=device,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_out_path = Path(args.skipped_out) if str(args.skipped_out or "").strip() else None
    if skipped_out_path is not None:
        skipped_out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = read_existing_output_ids([out_path, skipped_out_path]) if args.resume else set()
    output_mode = "a" if args.resume else "w"

    pending_tasks: List[EvidenceTask] = []
    wrote = 0
    skipped_existing = 0
    skipped_ties = 0
    skipped_margin = 0
    skipped_low_similarity = 0
    skipped_low_raw_similarity = 0
    with contextlib.ExitStack() as stack:
        handle = stack.enter_context(out_path.open(output_mode, encoding="utf-8", buffering=1))
        skipped_handle = (
            stack.enter_context(skipped_out_path.open(output_mode, encoding="utf-8", buffering=1))
            if skipped_out_path is not None
            else None
        )
        raw_score_pending_tasks: List[EvidenceTask] = []
        kept_tasks: List[EvidenceTask] = []
        raw_iterator = tqdm(
            iter_evidence_tasks(
                artifact_path=args.train_artifact,
                max_records=int(args.max_records),
            ),
            desc=f"dpo raw nodes {Path(args.train_artifact).name}",
            unit="node",
        )
        for task in raw_iterator:
            if task_output_id(task) in existing_ids:
                skipped_existing += 1
                raw_iterator.set_postfix(
                    wrote=wrote,
                    skipped_existing=skipped_existing,
                    skipped_ties=skipped_ties,
                    skipped_margin=skipped_margin,
                    skipped_low_similarity=skipped_low_similarity,
                    skipped_low_raw_similarity=skipped_low_raw_similarity,
                )
                continue
            raw_score_pending_tasks.append(task)
            if len(raw_score_pending_tasks) < int(args.raw_score_batch_size):
                continue
            batch_kept_tasks, batch_skipped_low_raw_similarity = filter_tasks_by_raw_query_score(
                tasks=raw_score_pending_tasks,
                skipped_handle=skipped_handle,
                reward_embedder_tokenizer=reward_embedder_tokenizer,
                reward_embedder_model=reward_embedder_model,
                device=device,
                reward_embedder_max_length=int(args.reward_embedder_max_length),
                min_raw_query_score=float(args.min_raw_query_score),
            )
            kept_tasks.extend(batch_kept_tasks)
            skipped_low_raw_similarity += batch_skipped_low_raw_similarity
            raw_score_pending_tasks = []
            raw_iterator.set_postfix(
                kept=len(kept_tasks),
                skipped_existing=skipped_existing,
                skipped_low_raw_similarity=skipped_low_raw_similarity,
            )
        if raw_score_pending_tasks:
            batch_kept_tasks, batch_skipped_low_raw_similarity = filter_tasks_by_raw_query_score(
                tasks=raw_score_pending_tasks,
                skipped_handle=skipped_handle,
                reward_embedder_tokenizer=reward_embedder_tokenizer,
                reward_embedder_model=reward_embedder_model,
                device=device,
                reward_embedder_max_length=int(args.reward_embedder_max_length),
                min_raw_query_score=float(args.min_raw_query_score),
            )
            kept_tasks.extend(batch_kept_tasks)
            skipped_low_raw_similarity += batch_skipped_low_raw_similarity

        iterator = tqdm(
            kept_tasks,
            desc=f"dpo gen nodes {Path(args.train_artifact).name}",
            unit="node",
        )
        for task in iterator:
            pending_tasks.append(task)
            if len(pending_tasks) < int(args.batch_size):
                continue
            prompts = [
                format_chat_prompt(generator_tokenizer, GENERATOR_SYSTEM_PROMPT, task.prompt)
                for task in pending_tasks
            ]
            raw_outputs = generate_candidate_rounds(
                tokenizer=generator_tokenizer,
                model=generator_model,
                prompts=prompts,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                num_rounds=int(args.num_rounds),
            )
            (
                batch_wrote,
                batch_skipped,
                batch_skipped_margin,
                batch_skipped_low_similarity,
            ) = write_batch(
                handle=handle,
                skipped_handle=skipped_handle,
                tasks=pending_tasks,
                prompts=prompts,
                raw_outputs=raw_outputs,
                reward_embedder_tokenizer=reward_embedder_tokenizer,
                reward_embedder_model=reward_embedder_model,
                device=device,
                min_reward_margin=float(args.min_reward_margin),
                min_chosen_query_score=float(args.min_chosen_query_score),
                reward_embedder_max_length=int(args.reward_embedder_max_length),
            )
            wrote += batch_wrote
            skipped_ties += batch_skipped
            skipped_margin += batch_skipped_margin
            skipped_low_similarity += batch_skipped_low_similarity
            pending_tasks = []
            iterator.set_postfix(
                wrote=wrote,
                skipped_existing=skipped_existing,
                skipped_ties=skipped_ties,
                skipped_margin=skipped_margin,
                skipped_low_similarity=skipped_low_similarity,
                skipped_low_raw_similarity=skipped_low_raw_similarity,
            )

        if pending_tasks:
            prompts = [
                format_chat_prompt(generator_tokenizer, GENERATOR_SYSTEM_PROMPT, task.prompt)
                for task in pending_tasks
            ]
            raw_outputs = generate_candidate_rounds(
                tokenizer=generator_tokenizer,
                model=generator_model,
                prompts=prompts,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                num_rounds=int(args.num_rounds),
            )
            (
                batch_wrote,
                batch_skipped,
                batch_skipped_margin,
                batch_skipped_low_similarity,
            ) = write_batch(
                handle=handle,
                skipped_handle=skipped_handle,
                tasks=pending_tasks,
                prompts=prompts,
                raw_outputs=raw_outputs,
                reward_embedder_tokenizer=reward_embedder_tokenizer,
                reward_embedder_model=reward_embedder_model,
                device=device,
                min_reward_margin=float(args.min_reward_margin),
                min_chosen_query_score=float(args.min_chosen_query_score),
                reward_embedder_max_length=int(args.reward_embedder_max_length),
            )
            wrote += batch_wrote
            skipped_ties += batch_skipped
            skipped_margin += batch_skipped_margin
            skipped_low_similarity += batch_skipped_low_similarity

    print(
        json.dumps(
            {
                "wrote": wrote,
                "skipped_existing": skipped_existing,
                "skipped_ties": skipped_ties,
                "skipped_margin": skipped_margin,
                "skipped_low_similarity": skipped_low_similarity,
                "skipped_low_raw_similarity": skipped_low_raw_similarity,
                "kept_after_raw_filter": len(kept_tasks),
                "min_raw_query_score": float(args.min_raw_query_score),
                "out": str(out_path),
                "skipped_out": str(skipped_out_path) if skipped_out_path is not None else "",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
