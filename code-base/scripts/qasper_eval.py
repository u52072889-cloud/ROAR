#!/usr/bin/env python
"""Answer and evidence evaluation helpers for QASPER-style records."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Sequence, Tuple


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_score_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLES_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_score_text(prediction).split()
    target_tokens = normalize_score_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    target_counts: Dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    common = 0
    for token in pred_tokens:
        count = target_counts.get(token, 0)
        if count <= 0:
            continue
        common += 1
        target_counts[token] = count - 1

    if common <= 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def exact_match(prediction: str, target: str) -> float:
    return 1.0 if normalize_score_text(prediction) == normalize_score_text(target) else 0.0


def best_answer_scores(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    if not references:
        return {"answer_f1": 0.0, "answer_em": 0.0}
    f1 = max(token_f1(prediction, ref) for ref in references)
    em = max(exact_match(prediction, ref) for ref in references)
    return {"answer_f1": float(f1), "answer_em": float(em)}


def set_f1(predicted: Iterable[int], gold: Iterable[int]) -> float:
    pred_set = {int(value) for value in predicted}
    gold_set = {int(value) for value in gold}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    common = len(pred_set & gold_set)
    if common <= 0:
        return 0.0
    precision = common / len(pred_set)
    recall = common / len(gold_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def best_evidence_scores(predicted: Iterable[int], gold_sets: Sequence[Sequence[int]]) -> Dict[str, float]:
    if not gold_sets:
        return {"evidence_f1": 0.0, "evidence_em": 0.0}
    best_f1 = 0.0
    best_em = 0.0
    pred_set = {int(value) for value in predicted}
    for gold in gold_sets:
        gold_set = {int(value) for value in gold}
        best_f1 = max(best_f1, set_f1(pred_set, gold_set))
        best_em = max(best_em, 1.0 if pred_set == gold_set else 0.0)
    return {"evidence_f1": float(best_f1), "evidence_em": float(best_em)}


def mean_metric(rows: Sequence[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)))


def aggregate_prediction_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {
        "answer_f1": mean_metric(rows, "answer_f1"),
        "answer_em": mean_metric(rows, "answer_em"),
        "evidence_f1": mean_metric(rows, "evidence_f1"),
        "evidence_em": mean_metric(rows, "evidence_em"),
        "count": float(len(rows)),
    }


def qasper_question_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record["paper_id"]), str(record["question_id"])
