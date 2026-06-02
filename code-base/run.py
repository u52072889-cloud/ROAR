#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path


DEFAULT_SEED_CANDIDATE_K = 12
DEFAULT_MAX_HOPS = 4
DEFAULT_BEAM_WIDTH = 6
DEFAULT_TOP_K = 16
DEFAULT_ROUTE_SEED_LIMIT = 16
CURRENT_PIPELINE_FIXED_SEED_EVIDENCE_COUNT = 5
CURRENT_PIPELINE_FRONTIER_SELECT_COUNT = 3
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_REWARD_EMBEDDER = "BAAI/bge-large-en-v1.5"
DEFAULT_SEED_EMBEDDER = "BAAI/bge-large-en-v1.5"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def seed_was_explicit(argv: list[str]) -> bool:
    return any(arg == "--seed" or arg.startswith("--seed=") for arg in argv)


def require_existing(paths: list[Path], *, stage: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage '{stage}' requires existing artifacts, but these files are missing: {missing}"
        )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def add_common_model_args(cmd: list[str], *, allow_download: bool, device: str) -> None:
    if device:
        cmd.extend(["--device", str(device)])
    if allow_download:
        cmd.append("--allow-download")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "components", "final", "experiment", "all"], default="all")
    parser.add_argument("--dataset", default="allenai/qasper")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=0)
    parser.add_argument("--max-section-chars", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true", default=True)
    parser.add_argument("--offline", dest="allow_download", action="store_false")

    parser.add_argument("--train-summary-data", default="")
    parser.add_argument("--eval-summary-data", default="")
    parser.add_argument("--train-pointer-data", default="")
    parser.add_argument("--eval-pointer-data", default="")
    parser.add_argument("--train-memory-data", default="")
    parser.add_argument("--eval-memory-data", default="")

    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--reward-embedder", default=DEFAULT_REWARD_EMBEDDER)
    parser.add_argument("--dpo-build-batch-size", type=int, default=4)
    parser.add_argument("--dpo-build-max-new-tokens", type=int, default=128)
    parser.add_argument("--dpo-temperature", type=float, default=0.8)
    parser.add_argument("--dpo-top-p", type=float, default=0.95)
    parser.add_argument("--dpo-build-max-records", type=int, default=0)
    parser.add_argument("--dpo-num-rounds", type=int, default=4)
    parser.add_argument("--dpo-min-reward-margin", type=float, default=0.10)
    parser.add_argument("--dpo-min-chosen-query-score", type=float, default=0.20)
    parser.add_argument("--dpo-min-raw-query-score", type=float, default=0.30)
    parser.add_argument("--dpo-reward-embedder-max-length", type=int, default=256)
    parser.add_argument("--dpo-epochs", type=float, default=1.0)
    parser.add_argument("--dpo-train-batch", type=int, default=1)
    parser.add_argument("--dpo-grad-accum", type=int, default=16)
    parser.add_argument("--dpo-lr", type=float, default=5e-5)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-train-max-records", type=int, default=0)
    parser.add_argument("--dpo-lora-r", type=int, default=8)
    parser.add_argument("--dpo-lora-alpha", type=int, default=32)
    parser.add_argument("--dpo-lora-dropout", type=float, default=0.05)
    parser.add_argument("--dpo-lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--dpo-infer-batch-size", type=int, default=8)
    parser.add_argument("--dpo-infer-max-new-tokens", type=int, default=128)
    parser.add_argument("--dpo-paper-summary-cache-size", type=int, default=1)
    parser.add_argument("--dpo-infer-max-records", type=int, default=0)
    parser.add_argument("--dpo-resume", action="store_true")

    parser.add_argument("--max-pointer-links", type=int, default=4)
    parser.add_argument("--structural-pointer-k", type=int, default=4)
    parser.add_argument("--pointer-candidate-k", type=int, default=0)
    parser.add_argument("--pointer-link-k", type=int, default=0)

    parser.add_argument("--seed-candidate-k", type=int, default=DEFAULT_SEED_CANDIDATE_K)
    parser.add_argument("--seed-embedder", default=DEFAULT_SEED_EMBEDDER)
    parser.add_argument("--seed-embedding-device", default="cuda")
    parser.add_argument("--seed-embedding-batch-size", type=int, default=64)
    parser.add_argument("--seed-embedding-max-length", type=int, default=256)

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--raw-max-len", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--route-seed-limit", type=int, default=DEFAULT_ROUTE_SEED_LIMIT)
    parser.add_argument("--max-evidence-nodes", type=int, default=8)
    parser.add_argument("--evidence-max-chars", type=int, default=0)
    parser.add_argument("--graph-head-hidden", type=int, default=256)
    parser.add_argument("--graph-head-dropout", type=float, default=0.1)
    parser.add_argument("--answer-router-loss-weight", type=float, default=0.05)
    parser.add_argument("--answer-router-baseline-momentum", type=float, default=0.95)
    parser.add_argument("--answer-router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--router-start-prior-weight", type=float, default=0.0)
    parser.add_argument("--router-edge-prior-weight", type=float, default=0.0)
    parser.add_argument("--fixed-seed-evidence-count", type=int, default=CURRENT_PIPELINE_FIXED_SEED_EVIDENCE_COUNT)
    parser.add_argument("--frontier-select-count", type=int, default=CURRENT_PIPELINE_FRONTIER_SELECT_COUNT)
    parser.add_argument("--frontier-hop-count", type=int, default=2)
    parser.add_argument("--frontier-baseline-seed-count", type=int, default=8)
    parser.add_argument("--frontier-max-per-seed", type=int, default=0)
    parser.add_argument("--query-conditioned-frontier-reranker", action="store_true")
    parser.add_argument("--disable-query-conditioned-frontier-reranker", dest="query_conditioned_frontier_reranker", action="store_false")
    parser.add_argument("--frontier-query-reranker-weight", type=float, default=2.0)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.add_argument("--disable-reader-summary-header", action="store_true")
    parser.add_argument(
        "--seed-only-baseline",
        action="store_true",
        help="For the final reader stage, use the top embedding seed nodes directly instead of graph routing.",
    )
    parser.add_argument(
        "--use-seed-evidence-only",
        action="store_true",
        help="Alias for --seed-only-baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(query_conditioned_frontier_reranker=True)
    args = parser.parse_args()
    if args.seed_only_baseline:
        args.use_seed_evidence_only = True

    if args.seed_candidate_k <= 0:
        raise ValueError("--seed-candidate-k must be positive.")
    for name in ["train_size", "eval_size"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if bool(args.train_summary_data) != bool(args.eval_summary_data):
        raise ValueError("--train-summary-data and --eval-summary-data must be provided together.")
    if bool(args.train_pointer_data) != bool(args.eval_pointer_data):
        raise ValueError("--train-pointer-data and --eval-pointer-data must be provided together.")
    if bool(args.train_memory_data) != bool(args.eval_memory_data):
        raise ValueError("--train-memory-data and --eval-memory-data must be provided together.")
    supplied_inputs = sum(bool(value) for value in [args.train_summary_data, args.train_pointer_data, args.train_memory_data])
    if supplied_inputs > 1:
        raise ValueError("--train-summary-data, --train-pointer-data, and --train-memory-data are mutually exclusive.")
    if args.max_hops <= 0 or args.beam_width <= 0 or args.route_seed_limit <= 0:
        raise ValueError("--max-hops, --beam-width, and --route-seed-limit must be positive.")
    if args.stage in {"final", "all", "experiment"}:
        if args.use_seed_evidence_only:
            if args.answer_router_loss_weight < 0.0:
                raise ValueError("--answer-router-loss-weight must be non-negative.")
        elif args.answer_router_loss_weight <= 0.0:
            raise ValueError("--answer-router-loss-weight must be positive for the current pipeline.")
        if args.answer_router_entropy_weight < 0.0:
            raise ValueError("--answer-router-entropy-weight must be non-negative.")
        if not 0.0 <= args.answer_router_baseline_momentum < 1.0:
            raise ValueError("--answer-router-baseline-momentum must be in [0, 1).")
        if args.router_start_prior_weight < 0.0 or args.router_edge_prior_weight < 0.0:
            raise ValueError("router prior weights must be non-negative.")
        if args.fixed_seed_evidence_count < 0 or args.frontier_select_count < 0:
            raise ValueError("frontier counts must be non-negative.")
        if args.frontier_hop_count <= 0:
            raise ValueError("--frontier-hop-count must be positive.")
        if args.frontier_baseline_seed_count < 0 or args.frontier_max_per_seed < 0:
            raise ValueError("frontier limits must be non-negative.")
        if args.frontier_query_reranker_weight < 0.0:
            raise ValueError("--frontier-query-reranker-weight must be non-negative.")
        total_frontier_nodes = args.fixed_seed_evidence_count + args.frontier_select_count
        if total_frontier_nodes > args.max_evidence_nodes:
            raise ValueError("--fixed-seed-evidence-count + --frontier-select-count must not exceed --max-evidence-nodes.")
        if total_frontier_nodes > args.top_k:
            raise ValueError("--fixed-seed-evidence-count + --frontier-select-count must not exceed --top-k.")
        if args.route_seed_limit < total_frontier_nodes:
            raise ValueError("--route-seed-limit must be at least --fixed-seed-evidence-count + --frontier-select-count.")
        if args.frontier_baseline_seed_count > 0 and args.frontier_baseline_seed_count < total_frontier_nodes:
            raise ValueError("--frontier-baseline-seed-count must be 0 or at least the selected evidence count.")
        if args.frontier_max_per_seed > 0 and args.frontier_select_count > args.fixed_seed_evidence_count * args.frontier_max_per_seed:
            raise ValueError("--frontier-select-count exceeds --fixed-seed-evidence-count * --frontier-max-per-seed.")
    if args.stage == "experiment" and not seed_was_explicit(sys.argv[1:]):
        args.seed = secrets.randbelow(2**31 - 1) + 1
        print(f"[experiment] randomized seed={args.seed}", flush=True)

    root = Path(__file__).resolve().parent
    scripts = root / "scripts"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or str(root / ".hf" / "hub")

    train_graph = artifacts / "qasper_train_graph.jsonl"
    eval_graph = artifacts / "qasper_test_graph.jsonl"
    train_raw = artifacts / "qasper_train_raw.jsonl"
    eval_raw = artifacts / "qasper_test_raw.jsonl"

    dpo_root = artifacts / "qasper_dpo"
    dpo_data = dpo_root / "data" / "qasper_train_router_dpo.jsonl"
    skipped_dpo_data = dpo_root / "data" / "qasper_train_router_dpo_skipped.jsonl"
    dpo_adapter = dpo_root / "checkpoints" / "generator_dpo_lora"
    train_summary = artifacts / "qasper_train_summary_dpo.jsonl"
    eval_summary = artifacts / "qasper_test_summary_dpo.jsonl"
    train_pointer = artifacts / "qasper_train_summary_dpo_pointer.jsonl"
    eval_pointer = artifacts / "qasper_test_summary_dpo_pointer.jsonl"
    train_linksum = artifacts / "qasper_train_summary_dpo_linksum.jsonl"
    eval_linksum = artifacts / "qasper_test_summary_dpo_linksum.jsonl"
    train_seeded = artifacts / "qasper_train_summary_dpo_linksum_seeded.jsonl"
    eval_seeded = artifacts / "qasper_test_summary_dpo_linksum_seeded.jsonl"
    raw_out = artifacts / f"raw_linear_qasper_seed{args.seed}"
    graph_out = artifacts / (
        f"qasper_seed_only_reader_seed{args.seed}"
        if args.use_seed_evidence_only
        else f"qasper_routed_reader_seed{args.seed}"
    )

    if args.stage == "components":
        require_existing([train_graph, eval_graph], stage="components")
    if args.stage == "final":
        require_existing([train_seeded, eval_seeded], stage="final")
    if args.stage == "experiment":
        if args.train_memory_data:
            require_existing([Path(args.train_memory_data), Path(args.eval_memory_data)], stage="experiment")
        elif args.train_pointer_data:
            require_existing([Path(args.train_pointer_data), Path(args.eval_pointer_data)], stage="experiment")
        elif args.train_summary_data:
            require_existing([Path(args.train_summary_data), Path(args.eval_summary_data)], stage="experiment")
        else:
            require_existing([train_summary, eval_summary], stage="experiment")

    if args.stage in {"prepare", "all"}:
        prepare_cmd = [
            sys.executable,
            str(scripts / "build_qasper_graph.py"),
            "--dataset", args.dataset,
            "--dataset-config", args.dataset_config,
            "--cache-dir", cache_dir,
            "--train-split", args.train_split,
            "--eval-split", args.eval_split,
            "--train-limit", str(args.train_size),
            "--eval-limit", str(args.eval_size),
            "--max-section-chars", str(args.max_section_chars),
            "--out-train", str(train_graph),
            "--out-eval", str(eval_graph),
            "--out-train-raw", str(train_raw),
            "--out-eval-raw", str(eval_raw),
        ]
        if args.allow_download:
            prepare_cmd.append("--allow-download")
        run(prepare_cmd)

    train_summary_source = Path(args.train_summary_data) if args.train_summary_data else train_summary
    eval_summary_source = Path(args.eval_summary_data) if args.eval_summary_data else eval_summary
    train_pointer_source = Path(args.train_pointer_data) if args.train_pointer_data else train_pointer
    eval_pointer_source = Path(args.eval_pointer_data) if args.eval_pointer_data else eval_pointer
    train_memory_source = Path(args.train_memory_data) if args.train_memory_data else train_linksum
    eval_memory_source = Path(args.eval_memory_data) if args.eval_memory_data else eval_linksum
    use_existing_default_summaries = args.stage == "experiment" and not args.train_summary_data

    if args.stage in {"components", "all", "experiment"}:
        if not args.train_memory_data:
            if not args.train_pointer_data:
                if not args.train_summary_data and not use_existing_default_summaries:
                    build_dpo_cmd = [
                        sys.executable,
                        str(scripts / "build_qasper_dpo_data.py"),
                        "--train-artifact", str(train_graph),
                        "--generator-model", args.generator_model,
                        "--reward-embedder", args.reward_embedder,
                        "--out", str(dpo_data),
                        "--skipped-out", str(skipped_dpo_data),
                        "--cache-dir", cache_dir,
                        "--batch-size", str(args.dpo_build_batch_size),
                        "--max-new-tokens", str(args.dpo_build_max_new_tokens),
                        "--temperature", str(args.dpo_temperature),
                        "--top-p", str(args.dpo_top_p),
                        "--max-records", str(args.dpo_build_max_records),
                        "--num-rounds", str(args.dpo_num_rounds),
                        "--min-reward-margin", str(args.dpo_min_reward_margin),
                        "--min-chosen-query-score", str(args.dpo_min_chosen_query_score),
                        "--min-raw-query-score", str(args.dpo_min_raw_query_score),
                        "--reward-embedder-max-length", str(args.dpo_reward_embedder_max_length),
                        "--seed", str(args.seed),
                    ]
                    if args.dpo_resume:
                        build_dpo_cmd.append("--resume")
                    add_common_model_args(build_dpo_cmd, allow_download=args.allow_download, device=args.generator_device)
                    run(build_dpo_cmd)

                    train_dpo_cmd = [
                        sys.executable,
                        str(scripts / "train_qasper_dpo.py"),
                        "--data", str(dpo_data),
                        "--model", args.generator_model,
                        "--out", str(dpo_adapter),
                        "--cache-dir", cache_dir,
                        "--epochs", str(args.dpo_epochs),
                        "--batch", str(args.dpo_train_batch),
                        "--grad-accum", str(args.dpo_grad_accum),
                        "--lr", str(args.dpo_lr),
                        "--beta", str(args.dpo_beta),
                        "--seed", str(args.seed),
                        "--max-records", str(args.dpo_train_max_records),
                        "--lora-r", str(args.dpo_lora_r),
                        "--lora-alpha", str(args.dpo_lora_alpha),
                        "--lora-dropout", str(args.dpo_lora_dropout),
                        "--lora-target-modules", args.dpo_lora_target_modules,
                    ]
                    add_common_model_args(train_dpo_cmd, allow_download=args.allow_download, device=args.generator_device)
                    run(train_dpo_cmd)

                    for data_path, out_path in [(train_graph, train_summary), (eval_graph, eval_summary)]:
                        infer_cmd = [
                            sys.executable,
                            str(scripts / "generate_qasper_dpo_summaries.py"),
                            "--data", str(data_path),
                            "--out", str(out_path),
                            "--model", args.generator_model,
                            "--adapter-path", str(dpo_adapter),
                            "--cache-dir", cache_dir,
                            "--batch-size", str(args.dpo_infer_batch_size),
                            "--max-new-tokens", str(args.dpo_infer_max_new_tokens),
                            "--paper-summary-cache-size", str(args.dpo_paper_summary_cache_size),
                            "--max-records", str(args.dpo_infer_max_records),
                        ]
                        if args.dpo_resume:
                            infer_cmd.append("--resume")
                        add_common_model_args(infer_cmd, allow_download=args.allow_download, device=args.generator_device)
                        run(infer_cmd)

                for data_path, out_path in [(train_summary_source, train_pointer), (eval_summary_source, eval_pointer)]:
                    run([
                        sys.executable,
                        str(scripts / "build_qasper_pointer_links.py"),
                        "--data", str(data_path),
                        "--out", str(out_path),
                        "--max-pointer-links", str(args.max_pointer_links),
                        "--structural-pointer-k", str(args.structural_pointer_k),
                        "--pointer-candidate-k", str(args.pointer_candidate_k),
                        "--pointer-link-k", str(args.pointer_link_k),
                    ])

            for data_path, out_path in [(train_pointer_source, train_linksum), (eval_pointer_source, eval_linksum)]:
                linksum_cmd = [
                    sys.executable,
                    str(scripts / "build_qasper_link_summaries.py"),
                    "--input", str(data_path),
                    "--output", str(out_path),
                    "--model", args.generator_model,
                    "--cache-dir", cache_dir,
                ]
                add_common_model_args(linksum_cmd, allow_download=args.allow_download, device=args.generator_device)
                run(linksum_cmd)

        for data_path, out_path in [(train_memory_source, train_seeded), (eval_memory_source, eval_seeded)]:
            seed_cmd = [
                sys.executable,
                str(scripts / "recompute_qasper_embedding_seed_candidates.py"),
                "--data", str(data_path),
                "--out", str(out_path),
                "--embedder", args.seed_embedder,
                "--cache-dir", cache_dir,
                "--batch-size", str(args.seed_embedding_batch_size),
                "--max-length", str(args.seed_embedding_max_length),
                "--seed-candidate-k", str(args.seed_candidate_k),
            ]
            add_common_model_args(seed_cmd, allow_download=args.allow_download, device=args.seed_embedding_device)
            run(seed_cmd)

    if args.stage in {"all"}:
        raw_cmd = [
            sys.executable,
            str(scripts / "train_qasper_raw_baseline.py"),
            "--data", str(train_raw),
            "--eval-data", str(eval_raw),
            "--model", args.model,
            "--cache-dir", cache_dir,
            "--out", str(raw_out),
            "--max-len", str(args.raw_max_len),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--seed", str(args.seed),
            "--gen-max-new-tokens", str(args.gen_max_new_tokens),
        ]
        if args.allow_download:
            raw_cmd.append("--allow-download")
        run(raw_cmd)

    if args.stage in {"final", "all", "experiment"}:
        answer_router_loss_weight = 0.0 if args.use_seed_evidence_only else float(args.answer_router_loss_weight)
        graph_cmd = [
            sys.executable,
            str(scripts / "train_qasper_routed_reader.py"),
            "--data", str(train_seeded),
            "--eval-data", str(eval_seeded),
            "--model", args.model,
            "--cache-dir", cache_dir,
            "--out", str(graph_out),
            "--max-len", str(args.max_len),
            "--top-k", str(args.top_k),
            "--max-hops", str(args.max_hops),
            "--beam-width", str(args.beam_width),
            "--max-evidence-nodes", str(args.max_evidence_nodes),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--graph-head-hidden", str(args.graph_head_hidden),
            "--graph-head-dropout", str(args.graph_head_dropout),
            "--route-seed-limit", str(args.route_seed_limit),
            "--answer-router-loss-weight", str(answer_router_loss_weight),
            "--answer-router-baseline-momentum", str(args.answer_router_baseline_momentum),
            "--answer-router-entropy-weight", str(args.answer_router_entropy_weight),
            "--router-start-prior-weight", str(args.router_start_prior_weight),
            "--router-edge-prior-weight", str(args.router_edge_prior_weight),
            "--fixed-seed-evidence-count", str(args.fixed_seed_evidence_count),
            "--frontier-select-count", str(args.frontier_select_count),
            "--frontier-hop-count", str(args.frontier_hop_count),
            "--frontier-baseline-seed-count", str(args.frontier_baseline_seed_count),
            "--frontier-max-per-seed", str(args.frontier_max_per_seed),
            "--frontier-query-reranker-weight", str(args.frontier_query_reranker_weight),
            "--gen-max-new-tokens", str(args.gen_max_new_tokens),
            "--seed", str(args.seed),
        ]
        if args.allow_download:
            graph_cmd.append("--allow-download")
        if args.disable_reader_summary_header:
            graph_cmd.append("--disable-reader-summary-header")
        if not args.query_conditioned_frontier_reranker:
            graph_cmd.append("--disable-query-conditioned-frontier-reranker")
        if args.use_seed_evidence_only:
            graph_cmd.append("--use-seed-evidence-only")
        save_json(
            graph_out / "experiment_config.json",
            {
                "run_phase1_args": vars(args),
                "run_phase1_argv": sys.argv,
                "train_command": graph_cmd,
                "paths": {
                    "train_graph": str(train_graph),
                    "eval_graph": str(eval_graph),
                    "dpo_data": str(dpo_data),
                    "dpo_adapter": str(dpo_adapter),
                    "train_summary": str(train_summary_source),
                    "eval_summary": str(eval_summary_source),
                    "train_pointer_source": str(train_pointer_source),
                    "eval_pointer_source": str(eval_pointer_source),
                    "train_memory_source": str(train_memory_source),
                    "eval_memory_source": str(eval_memory_source),
                    "train_seeded": str(train_seeded),
                    "eval_seeded": str(eval_seeded),
                    "output_dir": str(graph_out),
                },
            },
        )
        run(graph_cmd)


if __name__ == "__main__":
    main()
