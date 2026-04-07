# Patent Retrieval Evaluation Code Files

This document lists the code files used by the retrieval evaluation workflow
for comparing fine-tuned and base embedding models.

## 1. Pipeline Overview

1. Build query/candidate sets with citation positives and BM25-based negatives.
2. Run embedding retrieval evaluation for both models on the same data.
3. Export per-run metrics and per-query comparison artifacts.
4. Run multiple seeds and aggregate mean/std metrics.

## 2. Core Evaluation Files

### `patent/eval_retrieval_bm25.py`
- Role: thin entry wrapper.
- Behavior: imports and calls `main()` from
  `patent/topk_citation_retrieval_bm25_ipc_compare.py`.
- Why it exists: keeps the CLI entry stable while core logic evolves.

### `patent/topk_citation_retrieval_bm25_ipc_compare.py`
- Role: main single-run evaluation implementation.
- Key parts:
  - Dataset/query building (year window or citation CSV anchors).
  - Positive construction from self/other citations.
  - Hard negative construction:
    - same-IPC BM25 hard negatives,
    - same-IPC semantic/random distractors,
    - optional cross-IPC BM25 hard negatives.
  - Embedding inference with HF models.
  - Retrieval ranking and metrics calculation.
- Main outputs (under one run directory):
  - `queries.jsonl`
  - `candidates_global.jsonl`
  - `dataset_meta.json`
  - `per_query_comparison.csv`
  - `summary_topk_retrieval.json`
  - `summary_metrics_table.csv`

### `patent/eval_retrieval_bm25_multiseed.py`
- Role: multi-seed orchestrator for reproducible reporting.
- Behavior:
  - calls `patent/eval_retrieval_bm25.py` for each seed,
  - collects `summary_topk_retrieval.json` from each run,
  - computes mean/std for key metrics.
- Outputs:
  - `multiseed_summary.json`
  - `multiseed_metrics_table.csv`

## 3. Runner Scripts Used in Experiments

### `patent/run_topk_wait_gpu01.sh`
- Role: practical production runner on physical GPU 0/1.
- Features:
  - waits for GPU availability (idle or threshold-based),
  - runs 3 seeds (`42,43,44` by default),
  - runs two seeds in parallel then one seed,
  - performs final 3-seed aggregation.

### `patent/run_gap_ablation_matrix_gpu01.sh`
- Role: parameter matrix runner for gap analysis.
- Current matrix examples:
  - `q60_n8`, `q120_n8`, `q120_n16`, `q240_n16`.
- Output:
  - `matrix_delta_summary.csv` for quick cross-config comparison.

### `patent/run_topk_q120_n16_harder_gpu01.sh`
- Role: fixed harder protocol launcher.
- Default protocol:
  - `query_size=120`,
  - `hard_neg=16`,
  - includes same-IPC semantic/random distractors,
  - includes cross-IPC hard negatives.

## 4. Typical Entry Commands

Single run (core logic):

```bash
/data3/ljh/anaconda3/envs/ms-swift/bin/python patent/eval_retrieval_bm25.py \
  --finetuned-model-path <ft_ckpt> \
  --base-model-path <base_model> \
  --run-name <run_name> \
  ...
```

Multi-seed aggregate:

```bash
/data3/ljh/anaconda3/envs/ms-swift/bin/python patent/eval_retrieval_bm25_multiseed.py \
  --finetuned-model-path <ft_ckpt> \
  --base-model-path <base_model> \
  --seeds 42,43,44 \
  ...
```

GPU-wait helper runner:

```bash
cd patent && ./run_topk_wait_gpu01.sh
```

## 5. Notes for Reproducibility

- Keep `seed`, `query-size`, and hard-negative settings fixed when comparing runs.
- Use the same exclude list (`--exclude-pubno-file`) across experiments.
- Compare models on the same generated query/candidate artifacts per run.
