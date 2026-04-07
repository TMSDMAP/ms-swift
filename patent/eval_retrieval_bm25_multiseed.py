#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_seed_list(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("empty seed list")
    return out


def mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(st.mean(vals)), "std": float(st.stdev(vals))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BM25+IPC retrieval evaluation with multiple seeds and aggregate results.")

    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--runner", type=str, default="/home/ljh/data1/patent/eval_retrieval_bm25.py")

    p.add_argument("--finetuned-model-path", type=str, required=True)
    p.add_argument("--base-model-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="/home/ljh/data1/patent/eval_outputs")
    p.add_argument("--run-prefix", type=str, default="")

    p.add_argument("--seeds", type=str, default="42,43,44")

    # Forwarded args to eval_retrieval_bm25.py
    p.add_argument("--db-path", type=str, default="/home/ljh/data1/patent/patent.sqlite")
    p.add_argument("--table", type=str, default="patents")
    p.add_argument("--query-year", type=int, default=2020)
    p.add_argument("--window-years", type=int, default=3)
    p.add_argument("--query-size", type=int, default=500)
    p.add_argument("--query-pool-multiplier", type=int, default=25)
    p.add_argument("--max-query-scan", type=int, default=300000)
    p.add_argument("--query-source-citation-csv", type=str, default="")

    p.add_argument("--hard-neg-per-query", type=int, default=10)
    p.add_argument("--ipc-semantic-per-query", type=int, default=0)
    p.add_argument("--ipc-random-per-query", type=int, default=10)
    p.add_argument("--cross-ipc-hard-per-query", type=int, default=0)
    p.add_argument("--bm25-top-k", type=int, default=200)
    p.add_argument("--cross-ipc-bm25-top-k", type=int, default=600)
    p.add_argument("--bm25-pool-per-ipc", type=int, default=5000)
    p.add_argument("--global-bm25-pool-size", type=int, default=0)

    p.add_argument("--exclude-pubno-file", type=str, default="")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--rank-chunk-size", type=int, default=64)
    p.add_argument("--sql-fetch-batch", type=int, default=16384)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)

    if args.run_prefix:
        run_prefix = args.run_prefix
    else:
        run_prefix = f"retrieval_bm25_multiseed_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    root_dir = Path(args.output_dir) / run_prefix
    root_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for seed in seeds:
        run_name = f"{run_prefix}/seed_{seed}"
        log(f"Running seed={seed}, run_name={run_name}")

        cmd = [
            args.python_bin,
            args.runner,
            "--db-path",
            args.db_path,
            "--table",
            args.table,
            "--finetuned-model-path",
            args.finetuned_model_path,
            "--base-model-path",
            args.base_model_path,
            "--output-dir",
            args.output_dir,
            "--run-name",
            run_name,
            "--query-year",
            str(args.query_year),
            "--window-years",
            str(args.window_years),
            "--query-size",
            str(args.query_size),
            "--query-pool-multiplier",
            str(args.query_pool_multiplier),
            "--max-query-scan",
            str(args.max_query_scan),
            "--hard-neg-per-query",
            str(args.hard_neg_per_query),
            "--ipc-semantic-per-query",
            str(args.ipc_semantic_per_query),
            "--ipc-random-per-query",
            str(args.ipc_random_per_query),
            "--cross-ipc-hard-per-query",
            str(args.cross_ipc_hard_per_query),
            "--bm25-top-k",
            str(args.bm25_top_k),
            "--cross-ipc-bm25-top-k",
            str(args.cross_ipc_bm25_top_k),
            "--bm25-pool-per-ipc",
            str(args.bm25_pool_per_ipc),
            "--global-bm25-pool-size",
            str(args.global_bm25_pool_size),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--rank-chunk-size",
            str(args.rank_chunk_size),
            "--sql-fetch-batch",
            str(args.sql_fetch_batch),
            "--seed",
            str(seed),
        ]

        if args.exclude_pubno_file:
            cmd.extend(["--exclude-pubno-file", args.exclude_pubno_file])
        if args.query_source_citation_csv:
            cmd.extend(["--query-source-citation-csv", args.query_source_citation_csv])

        subprocess.run(cmd, check=True)

        summary_path = Path(args.output_dir) / run_name / "summary_topk_retrieval.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary not found: {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "seed": seed,
                "run_name": run_name,
                "summary_path": str(summary_path),
                "metrics": summary["metrics"],
                "paired_comparison": summary.get("paired_comparison", {}),
            }
        )

    metric_keys = [
        "mean_rank",
        "mrr",
        "map",
        "ndcg",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "recall@20",
    ]

    agg = {
        "finetuned": {},
        "base": {},
        "delta_ft_minus_base": {},
    }

    for k in metric_keys:
        ft_vals = [float(r["metrics"]["finetuned"][k]) for r in runs]
        base_vals = [float(r["metrics"]["base"][k]) for r in runs]
        delta_vals = [a - b for a, b in zip(ft_vals, base_vals)]

        agg["finetuned"][k] = mean_std(ft_vals)
        agg["base"][k] = mean_std(base_vals)
        agg["delta_ft_minus_base"][k] = mean_std(delta_vals)

    out = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_prefix": run_prefix,
        "seeds": seeds,
        "runner": args.runner,
        "runs": runs,
        "aggregate": agg,
    }

    out_json = root_dir / "multiseed_summary.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = root_dir / "multiseed_metrics_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "finetuned_mean", "finetuned_std", "base_mean", "base_std", "delta_mean", "delta_std"])
        for k in metric_keys:
            writer.writerow(
                [
                    k,
                    agg["finetuned"][k]["mean"],
                    agg["finetuned"][k]["std"],
                    agg["base"][k]["mean"],
                    agg["base"][k]["std"],
                    agg["delta_ft_minus_base"][k]["mean"],
                    agg["delta_ft_minus_base"][k]["std"],
                ]
            )

    print(json.dumps({"multiseed_summary": str(out_json), "multiseed_table": str(out_csv)}, ensure_ascii=False, indent=2))
    print(f"[DONE] multiseed outputs saved to: {root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
