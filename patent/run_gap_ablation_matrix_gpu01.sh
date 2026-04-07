#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljh/data1/patent"
RUNNER="${ROOT}/run_topk_wait_gpu01.sh"

MATRIX_PREFIX="${MATRIX_PREFIX:-bm25_gap_matrix_$(date +%Y%m%d-%H%M%S)}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-50000}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-20}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44}"

MATRIX_OUT="${ROOT}/eval_outputs/${MATRIX_PREFIX}"
mkdir -p "${MATRIX_OUT}"
MATRIX_LOG="${MATRIX_OUT}/matrix.log"

log() {
  local msg="$*"
  echo "[$(date '+%F %T')] ${msg}" | tee -a "${MATRIX_LOG}" >&2
}

run_one() {
  local label="$1"
  local qsize="$2"
  local hard_neg="$3"
  local rand_neg="$4"

  local run_prefix="${MATRIX_PREFIX}_${label}_s3_gpu01"

  log "Start config=${label}: query_size=${qsize}, hard_neg=${hard_neg}, ipc_random_neg=${rand_neg}"

  RUN_PREFIX="${run_prefix}" \
  QUERY_SIZE="${qsize}" \
  HARD_NEG_PER_QUERY="${hard_neg}" \
  IPC_RANDOM_PER_QUERY="${rand_neg}" \
  IPC_SEMANTIC_PER_QUERY=0 \
  CROSS_IPC_HARD_PER_QUERY=0 \
  BM25_TOP_K=80 \
  CROSS_IPC_BM25_TOP_K=0 \
  BM25_POOL_PER_IPC=300 \
  GLOBAL_BM25_POOL_SIZE=0 \
  WAIT_SECONDS="${WAIT_SECONDS}" \
  FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB}" \
  UTIL_THRESHOLD="${UTIL_THRESHOLD}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  "${RUNNER}"

  local table_path="${ROOT}/eval_outputs/${run_prefix}/multiseed_metrics_table.csv"
  if [[ ! -f "${table_path}" ]]; then
    log "FAILED config=${label}: missing ${table_path}"
    exit 1
  fi

  log "Done config=${label}: ${table_path}"
}

build_matrix_summary() {
  log "Building matrix delta summary CSV"

  /data3/ljh/anaconda3/envs/ms-swift/bin/python - "${MATRIX_OUT}" "${MATRIX_PREFIX}" <<'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
prefix = sys.argv[2]

configs = [
    ("q60_n8", f"{prefix}_q60_n8_s3_gpu01"),
    ("q120_n8", f"{prefix}_q120_n8_s3_gpu01"),
    ("q120_n16", f"{prefix}_q120_n16_s3_gpu01"),
    ("q240_n16", f"{prefix}_q240_n16_s3_gpu01"),
]

wanted = {"mean_rank", "mrr", "map", "ndcg", "hit@1", "recall@10"}
rows = []

for label, run_prefix in configs:
    table = Path("/home/ljh/data1/patent/eval_outputs") / run_prefix / "multiseed_metrics_table.csv"
    if not table.exists():
        raise FileNotFoundError(table)

    with table.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            metric = r["metric"]
            if metric not in wanted:
                continue
            rows.append(
                {
                    "config": label,
                    "run_prefix": run_prefix,
                    "metric": metric,
                    "delta_mean": r["delta_mean"],
                    "delta_std": r["delta_std"],
                    "finetuned_mean": r["finetuned_mean"],
                    "base_mean": r["base_mean"],
                }
            )

out_csv = out_dir / "matrix_delta_summary.csv"
with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "config",
            "run_prefix",
            "metric",
            "delta_mean",
            "delta_std",
            "finetuned_mean",
            "base_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(str(out_csv))
PY
}

main() {
  log "Matrix prefix=${MATRIX_PREFIX}, seeds=${SEEDS_CSV}"

  run_one "q60_n8" 60 8 8
  run_one "q120_n8" 120 8 8
  run_one "q120_n16" 120 16 16
  run_one "q240_n16" 240 16 16

  build_matrix_summary
  log "DONE matrix run. Outputs root=${MATRIX_OUT}"
}

main "$@"
