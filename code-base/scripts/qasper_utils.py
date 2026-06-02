#!/usr/bin/env python
"""QASPER-specific dataset and record helpers."""

from __future__ import annotations

import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen

from datasets import load_dataset

from data_utils import compact_whitespace


FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED"
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[1] / ".hf" / "hub")
QASPER_CACHE_SUBDIR = "qasper_train_dev"
QASPER_TRAIN_DEV_URL = "https://qasper-dataset.s3-us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST_URL = "https://qasper-dataset.s3-us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
QASPER_SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "eval": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}
QASPER_SPLIT_DOWNLOADS = {
    "train": (QASPER_TRAIN_DEV_URL, ("qasper-train-v0.3.json", "qasper-dev-v0.3.json")),
    "validation": (QASPER_TRAIN_DEV_URL, ("qasper-train-v0.3.json", "qasper-dev-v0.3.json")),
    "dev": (QASPER_TRAIN_DEV_URL, ("qasper-train-v0.3.json", "qasper-dev-v0.3.json")),
    "eval": (QASPER_TRAIN_DEV_URL, ("qasper-train-v0.3.json", "qasper-dev-v0.3.json")),
    "test": (QASPER_TEST_URL, ("qasper-test-v0.3.json",)),
}

PLACEHOLDER_SECTION_TITLE_RE = re.compile(r"(?:(?:::+)(?:\s*:::+)*)|Section \d+")
NUMBERING_ONLY_RE = re.compile(r"^\d+(?:\.\d+)+(?:[A-Za-z]+)?$")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def is_routable_paragraph(text: str, *, section_title: str) -> bool:
    paragraph_text = compact_whitespace(text)
    if not paragraph_text:
        return False

    title = compact_whitespace(section_title)
    placeholder_title = bool(PLACEHOLDER_SECTION_TITLE_RE.fullmatch(title))
    token_count = len(paragraph_text.split())
    lower_text = paragraph_text.lower()

    if NUMBERING_ONLY_RE.fullmatch(paragraph_text):
        return False
    if EMAIL_RE.search(paragraph_text):
        return False
    if paragraph_text.startswith(("— ", "- ")) and token_count <= 8:
        return False
    if placeholder_title and "$^{" in paragraph_text and token_count <= 40:
        return False
    if (
        placeholder_title
        and token_count <= 24
        and any(phrase in lower_text for phrase in ("department of", "university", "institute", "faculty of", "school of"))
        and not any(mark in paragraph_text for mark in ".!?;:")
    ):
        return False
    if placeholder_title and token_count <= 16 and paragraph_text.count(",") >= 4 and not any(mark in paragraph_text for mark in ".!?;:"):
        return False
    return True


@dataclass(frozen=True)
class AnswerReference:
    annotation_id: str
    worker_id: str
    answer_type: str
    normalized_answer: str
    evidence_node_ids: Tuple[int, ...]
    has_text_evidence: bool


def resolve_cached_qasper_path(
    *,
    split: str,
    cache_dir: str,
) -> str:
    split = str(split or "").lower()
    filenames = [QASPER_SPLIT_FILES.get(split, f"qasper-{split}-v0.3.json")]
    roots = [
        Path(cache_dir or DEFAULT_CACHE_DIR),
        Path(DEFAULT_CACHE_DIR),
    ]
    subdirs = [
        Path("."),
        Path(QASPER_CACHE_SUBDIR),
        Path("qasper"),
        Path("data"),
    ]
    for root in roots:
        for subdir in subdirs:
            for filename in filenames:
                candidate = root / subdir / filename
                if candidate.exists():
                    return str(candidate)
    return ""


def download_qasper_raw_files(
    *,
    split: str,
    cache_dir: str,
) -> str:
    normalized_split = str(split or "").lower()
    archive_info = QASPER_SPLIT_DOWNLOADS.get(normalized_split)
    if archive_info is None:
        raise RuntimeError(
            f"Unsupported QASPER split {split!r}. Pass a local --dataset path for custom splits."
        )

    archive_url, expected_files = archive_info
    target_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / QASPER_CACHE_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_name = Path(urlsplit(archive_url).path).name or f"qasper-{normalized_split}.tgz"
    archive_path = target_dir / archive_name

    if not archive_path.exists():
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            with urlopen(archive_url, timeout=120) as response, open(partial_path, "wb") as f_out:
                shutil.copyfileobj(response, f_out)
            partial_path.replace(archive_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    remaining = {name for name in expected_files if not (target_dir / name).exists()}
    if remaining:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                basename = Path(member.name).name
                if basename not in remaining:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, open(target_dir / basename, "wb") as f_out:
                    shutil.copyfileobj(source, f_out)
                remaining.remove(basename)
                if not remaining:
                    break
        if remaining:
            missing = ", ".join(sorted(remaining))
            raise RuntimeError(
                f"Downloaded {archive_name} but could not extract expected QASPER files: {missing}."
            )

    resolved_path = resolve_cached_qasper_path(split=normalized_split, cache_dir=cache_dir)
    if not resolved_path:
        expected_name = QASPER_SPLIT_FILES[normalized_split]
        raise RuntimeError(
            f"Downloaded QASPER archive for split {normalized_split!r}, but {expected_name} was not found under {target_dir}."
        )
    return resolved_path


def load_qasper_rows(
    dataset_name: str,
    *,
    split: str,
    dataset_config: str = "",
    cache_dir: str = "",
    allow_download: bool = False,
) -> Sequence[Dict[str, Any]]:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if dataset_name == "allenai/qasper":
        cached_path = resolve_cached_qasper_path(split=split, cache_dir=cache_dir)
        if cached_path:
            return load_qasper_rows(
                cached_path,
                split=split,
                dataset_config=dataset_config,
                cache_dir=cache_dir,
                allow_download=allow_download,
            )
        if allow_download:
            downloaded_path = download_qasper_raw_files(split=split, cache_dir=cache_dir)
            return load_qasper_rows(
                downloaded_path,
                split=split,
                dataset_config=dataset_config,
                cache_dir=cache_dir,
                allow_download=allow_download,
            )
    if Path(dataset_name).exists():
        suffix = Path(dataset_name).suffix.lower()
        if suffix == ".json":
            with open(dataset_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                rows: List[Dict[str, Any]] = []
                for paper_id, paper in data.items():
                    if not isinstance(paper, dict):
                        continue
                    row = dict(paper)
                    row.setdefault("id", str(paper_id))
                    row.setdefault("paper_id", str(paper_id))
                    rows.append(row)
                return rows
            if isinstance(data, list):
                return list(data)
        if suffix == ".jsonl":
            rows = []
            with open(dataset_name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    else:
        try:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        except RuntimeError as exc:
            if dataset_name == "allenai/qasper" and "Dataset scripts are no longer supported" in str(exc):
                raise RuntimeError(
                    "Failed to load allenai/qasper via datasets because script-based datasets are disabled in this environment. "
                    "Rerun with --allow-download, pass --dataset /path/to/qasper-train-v0.3.json, or place the raw files under "
                    f"{cache_dir}/qasper_train_dev/."
                ) from exc
            raise
    return dataset[split]


def normalize_answer_text(answer: Dict[str, Any]) -> Tuple[str, str]:
    if bool(answer.get("unanswerable")):
        return "unanswerable", "unanswerable"

    extractive_spans = [compact_whitespace(span) for span in (answer.get("extractive_spans") or []) if compact_whitespace(span)]
    if extractive_spans:
        unique_spans: List[str] = []
        seen = set()
        for span in extractive_spans:
            key = span.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_spans.append(span)
        return " ; ".join(unique_spans), "extractive"

    free_form_answer = compact_whitespace(answer.get("free_form_answer") or "")
    if free_form_answer:
        return free_form_answer, "free_form"

    yes_no = answer.get("yes_no")
    if yes_no is True:
        return "yes", "yes_no"
    if yes_no is False:
        return "no", "yes_no"

    return "unanswerable", "unanswerable"


def normalize_evidence_texts(answer: Dict[str, Any]) -> List[str]:
    text_evidence: List[str] = []
    seen_text = set()
    for raw in answer.get("evidence") or []:
        value = compact_whitespace(raw)
        if not value:
            continue
        if value.startswith(FLOAT_EVIDENCE_PREFIX):
            continue
        key = value.lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        text_evidence.append(value)
    return text_evidence


def safe_section_title(value: str, index: int) -> str:
    title = compact_whitespace(value)
    if title:
        return title
    return f"Section {index + 1}"


def iter_full_text_sections(paper: Dict[str, Any]) -> Iterator[Tuple[int, str, List[str]]]:
    full_text = paper.get("full_text") or []
    if isinstance(full_text, list):
        for section_index, item in enumerate(full_text):
            if not isinstance(item, dict):
                continue
            title = safe_section_title(str(item.get("section_name") or ""), section_index)
            paragraphs = [str(paragraph) for paragraph in (item.get("paragraphs") or [])]
            yield section_index, title, paragraphs
        return

    if isinstance(full_text, dict):
        section_names = list(full_text.get("section_name") or [])
        section_paragraphs = list(full_text.get("paragraphs") or [])
        total_sections = max(len(section_names), len(section_paragraphs))
        for section_index in range(total_sections):
            title = safe_section_title(section_names[section_index] if section_index < len(section_names) else "", section_index)
            paragraphs = list(section_paragraphs[section_index] or []) if section_index < len(section_paragraphs) else []
            yield section_index, title, [str(paragraph) for paragraph in paragraphs]


def iter_raw_full_text_sections(paper: Dict[str, Any]) -> Iterator[Tuple[str, List[str]]]:
    full_text = paper.get("full_text") or []
    if isinstance(full_text, list):
        for item in full_text:
            if not isinstance(item, dict):
                continue
            title = str(item.get("section_name") or "")
            paragraphs = [str(paragraph) for paragraph in (item.get("paragraphs") or [])]
            yield title, paragraphs
        return

    if isinstance(full_text, dict):
        section_names = list(full_text.get("section_name") or [])
        section_paragraphs = list(full_text.get("paragraphs") or [])
        total_sections = max(len(section_names), len(section_paragraphs))
        for section_index in range(total_sections):
            title = str(section_names[section_index] if section_index < len(section_names) else "")
            paragraphs = list(section_paragraphs[section_index] or []) if section_index < len(section_paragraphs) else []
            yield title, [str(paragraph) for paragraph in paragraphs]


def render_raw_document_text(paper: Dict[str, Any]) -> str:
    parts: List[str] = []
    for section_title, paragraphs in iter_raw_full_text_sections(paper):
        if section_title:
            parts.append(section_title)
        parts.extend(paragraphs)
    return "\n\n".join(parts)


def iter_questions(paper: Dict[str, Any]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    qas = paper.get("qas") or []
    if isinstance(qas, list):
        for question_index, item in enumerate(qas):
            if not isinstance(item, dict):
                continue
            yield question_index, dict(item)
        return

    if isinstance(qas, dict):
        questions = list(qas.get("question") or [])
        question_ids = list(qas.get("question_id") or [])
        answer_bundles = list(qas.get("answers") or [])
        for question_index, question in enumerate(questions):
            yield question_index, {
                "question": question,
                "question_id": question_ids[question_index] if question_index < len(question_ids) else f"q{question_index}",
                "answers": answer_bundles[question_index] if question_index < len(answer_bundles) else {},
            }


def iter_answers(question: Dict[str, Any], question_id: str) -> Iterator[Tuple[int, str, str, Dict[str, Any]]]:
    answers = question.get("answers") or []
    if isinstance(answers, list):
        for answer_index, item in enumerate(answers):
            if not isinstance(item, dict):
                continue
            annotation_id = str(item.get("annotation_id") or f"{question_id}_a{answer_index}")
            worker_id = str(item.get("worker_id") or "")
            answer_payload = item.get("answer") or {}
            if not isinstance(answer_payload, dict):
                answer_payload = {}
            yield answer_index, annotation_id, worker_id, answer_payload
        return

    if isinstance(answers, dict):
        answer_items = list(answers.get("answer") or [])
        annotation_ids = list(answers.get("annotation_id") or [])
        worker_ids = list(answers.get("worker_id") or [])
        for answer_index, answer_item in enumerate(answer_items):
            annotation_id = str(annotation_ids[answer_index] if answer_index < len(annotation_ids) else f"{question_id}_a{answer_index}")
            worker_id = str(worker_ids[answer_index] if answer_index < len(worker_ids) else "")
            payload = answer_item if isinstance(answer_item, dict) else {}
            yield answer_index, annotation_id, worker_id, payload


def build_paper_nodes(
    paper: Dict[str, Any],
    *,
    max_section_chars: int,
) -> List[Dict[str, Any]]:
    if int(max_section_chars) > 0:
        raise ValueError(
            f"max_section_chars={max_section_chars} would truncate section text. "
            "No truncation is allowed in this pipeline."
        )
    sections = list(iter_full_text_sections(paper))
    nodes: List[Dict[str, Any]] = []
    next_node_id = 1
    paragraph_count_total = sum(
        1
        for _section_index, _title, paragraphs in sections
        for paragraph in (paragraphs or [])
        if is_routable_paragraph(paragraph, section_title=_title)
    )
    running_paragraph_rank = 0

    for section_index, title, paragraphs in sections:
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_text = compact_whitespace(paragraph)
            if not is_routable_paragraph(paragraph_text, section_title=title):
                continue
            node_id = next_node_id
            next_node_id += 1
            running_paragraph_rank += 1
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": "paragraph",
                    "section_index": section_index,
                    "paragraph_index": paragraph_index,
                    "section_title": title,
                    "text": paragraph_text,
                    "position_features": {
                        "paragraph_frac": float(running_paragraph_rank - 1) / max(1, paragraph_count_total - 1) if paragraph_count_total > 1 else 0.0,
                    },
                    "token_hint": max(1, len(paragraph_text.split())),
                }
            )

    return nodes


def paragraph_text_index(nodes: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for node in nodes:
        if node.get("node_type") != "paragraph":
            continue
        normalized = compact_whitespace(node.get("text") or "").lower()
        if not normalized:
            continue
        index.setdefault(normalized, []).append(int(node["node_id"]))
    return index


def map_evidence_to_node_ids(
    evidence_texts: Sequence[str],
    nodes: Sequence[Dict[str, Any]],
) -> List[int]:
    paragraph_index = paragraph_text_index(nodes)
    paragraphs = [
        (int(node["node_id"]), compact_whitespace(node.get("text") or "").lower())
        for node in nodes
        if node.get("node_type") == "paragraph"
    ]

    matched_ids: List[int] = []
    seen = set()

    for raw in evidence_texts:
        evidence = compact_whitespace(raw)
        if not evidence:
            continue
        normalized = evidence.lower()
        candidates = paragraph_index.get(normalized)
        found: List[int] = []
        if candidates:
            found = list(candidates)
        else:
            for node_id, paragraph_text in paragraphs:
                if not paragraph_text:
                    continue
                if normalized in paragraph_text or paragraph_text in normalized:
                    found.append(node_id)
            if not found:
                best_score = 0.0
                best_ids: List[int] = []
                evidence_tokens = set(normalized.split())
                for node_id, paragraph_text in paragraphs:
                    paragraph_tokens = set(paragraph_text.split())
                    if not evidence_tokens or not paragraph_tokens:
                        continue
                    overlap = len(evidence_tokens & paragraph_tokens) / max(1, len(evidence_tokens | paragraph_tokens))
                    if overlap < 0.6:
                        continue
                    if overlap > best_score:
                        best_score = overlap
                        best_ids = [node_id]
                    elif overlap == best_score:
                        best_ids.append(node_id)
                found = best_ids
        if not found:
            continue
        for node_id in found:
            if node_id in seen:
                continue
            seen.add(node_id)
            matched_ids.append(node_id)

    return matched_ids


def render_linearized_document(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = compact_whitespace(record.get("doc_meta", {}).get("title") or "")
    abstract = compact_whitespace(record.get("doc_meta", {}).get("abstract") or "")
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    section_buffer: Dict[int, List[str]] = {}
    section_titles: Dict[int, str] = {}
    for node in record.get("nodes", []):
        if node.get("node_type") != "paragraph":
            continue
        section_index = int(node["section_index"])
        section_titles.setdefault(section_index, str(node.get("section_title") or ""))
        section_buffer.setdefault(section_index, []).append(str(node.get("text") or ""))

    for section_index in sorted(section_titles):
        title = compact_whitespace(section_titles[section_index])
        if title:
            parts.append(f"Section: {title}")
        else:
            parts.append(f"Section {section_index + 1}")
        for paragraph in section_buffer.get(section_index, []):
            paragraph = compact_whitespace(paragraph)
            if paragraph:
                parts.append(paragraph)

    return "\n\n".join(part for part in parts if part)


def question_records_for_paper(
    paper: Dict[str, Any],
    *,
    split: str,
    nodes: Sequence[Dict[str, Any]],
    keep_unanswerable: bool = True,
) -> Iterator[Dict[str, Any]]:
    title = compact_whitespace(paper.get("title") or "")
    abstract = compact_whitespace(paper.get("abstract") or "")
    paper_id = str(paper.get("id") or paper.get("paper_id") or "")

    for question_index, question_item in iter_questions(paper):
        question = compact_whitespace(question_item.get("question") or "")
        question_id = str(question_item.get("question_id") or f"q{question_index}")

        references: List[AnswerReference] = []
        for _answer_index, annotation_id, worker_id, answer_item in iter_answers(question_item, question_id):
            normalized_answer, answer_type = normalize_answer_text(answer_item)
            evidence_texts = normalize_evidence_texts(answer_item)
            evidence_node_ids = map_evidence_to_node_ids(evidence_texts, nodes)
            references.append(
                AnswerReference(
                    annotation_id=annotation_id,
                    worker_id=worker_id,
                    answer_type=answer_type,
                    normalized_answer=normalized_answer,
                    evidence_node_ids=tuple(evidence_node_ids),
                    has_text_evidence=bool(evidence_node_ids),
                )
            )

        answer_refs = [
            {
                "annotation_id": ref.annotation_id,
                "worker_id": ref.worker_id,
                "answer_type": ref.answer_type,
                "normalized_answer": ref.normalized_answer,
                "evidence_node_ids": list(ref.evidence_node_ids),
                "has_text_evidence": ref.has_text_evidence,
            }
            for ref in references
        ]

        for ref in references:
            if ref.answer_type == "unanswerable" and not keep_unanswerable:
                continue
            yield {
                "id": f"{paper_id}::{question_id}::{ref.annotation_id}",
                "paper_id": paper_id,
                "question_id": question_id,
                "annotation_id": ref.annotation_id,
                "doc_meta": {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                },
                "query": {
                    "question": question,
                    "question_index": question_index,
                },
                "target_answer": {
                    "answer_type": ref.answer_type,
                    "normalized_answer": ref.normalized_answer,
                },
                "answer_refs": answer_refs,
                "gold_evidence_node_ids": list(ref.evidence_node_ids),
                "split": split,
            }
