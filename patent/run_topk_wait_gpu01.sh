#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljh/data1/patent"
PY_BIN="/data3/ljh/anaconda3/envs/ms-swift/bin/python"
RUN_PREFIX="${RUN_PREFIX:-bm25_multiseed_harder_$(date +%Y%m%d)_q120_n16_s3_gpu01_wait}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-50000}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-20}"

SEEDS_CSV="${SEEDS_CSV:-42,43,44}"
QUERY_YEAR="${QUERY_YEAR:-2021}"
WINDOW_YEARS="${WINDOW_YEARS:-18}"
QUERY_SIZE="${QUERY_SIZE:-120}"
QUERY_POOL_MULTIPLIER="${QUERY_POOL_MULTIPLIER:-25}"
MAX_QUERY_SCAN="${MAX_QUERY_SCAN:-300000}"
HARD_NEG_PER_QUERY="${HARD_NEG_PER_QUERY:-16}"
IPC_SEMANTIC_PER_QUERY="${IPC_SEMANTIC_PER_QUERY:-8}"
IPC_RANDOM_PER_QUERY="${IPC_RANDOM_PER_QUERY:-16}"
CROSS_IPC_HARD_PER_QUERY="${CROSS_IPC_HARD_PER_QUERY:-8}"
BM25_TOP_K="${BM25_TOP_K:-120}"
CROSS_IPC_BM25_TOP_K="${CROSS_IPC_BM25_TOP_K:-1200}"
BM25_POOL_PER_IPC="${BM25_POOL_PER_IPC:-600}"
GLOBAL_BM25_POOL_SIZE="${GLOBAL_BM25_POOL_SIZE:-60000}"

PHYSICAL_GPU_A="${PHYSICAL_GPU_A:-0}"
PHYSICAL_GPU_B="${PHYSICAL_GPU_B:-1}"

FINETUNED_MODEL_PATH="${FINETUNED_MODEL_PATH:-/home/ljh/data1/output_backup/Qwen3_10W/v14-20260331-153258_independent_20260401-190135/checkpoint-700}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/ljh/data1/Qwen3-embedding-4B}"
QUERY_SOURCE_CITATION_CSV="${QUERY_SOURCE_CITATION_CSV:-${ROOT}/eval_outputs/benchmark_20260405-105001/citation_quartet_scores.csv}"
EXCLUDE_PUBNO_FILE="${EXCLUDE_PUBNO_FILE:-${ROOT}/eval_outputs/exclude_pubnos_trainval_full_20260405.txt}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-768}"
RANK_CHUNK_SIZE="${RANK_CHUNK_SIZE:-64}"
SQL_FETCH_BATCH="${SQL_FETCH_BATCH:-16384}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda:0}"

OUTPUT_DIR="${ROOT}/eval_outputs"
RUN_DIR="${OUTPUT_DIR}/${RUN_PREFIX}"
mkdir -p "${RUN_DIR}"
LAUNCHER_LOG="${RUN_DIR}/launcher.log"

log() {
  local msg="$*"
  echo "[$(date '+%F %T')] ${msg}" | tee -a "${LAUNCHER_LOG}" >&2
}

parse_seeds() {
  local raw="$1"
  IFS=',' read -r -a SEEDS <<< "${raw}"
  if [[ "${#SEEDS[@]}" -eq 0 ]]; then
    echo "SEEDS_CSV is empty" >&2
    exit 1
  fi
  for s in "${SEEDS[@]}"; do
    if ! [[ "${s}" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed value: ${s}" >&2
      exit 1
    fi
  done
}

gpu_pid_count() {
  local gpu_idx="$1"
  nvidia-smi -i "${gpu_idx}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | grep -Eo '^[0-9]+' \
    | wc -l
}

gpu_free_mem() {
  local gpu_idx="$1"
  nvidia-smi -i "${gpu_idx}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' '
}

gpu_util() {
  local gpu_idx="$1"
  nvidia-smi -i "${gpu_idx}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' '
}

wait_for_gpu_idle() {
  log "Waiting for physical GPU ${PHYSICAL_GPU_A} and ${PHYSICAL_GPU_B} to become idle/available..."
  log "Availability thresholds: free_mem>=${FREE_MEM_THRESHOLD_MB}MiB, util<=${UTIL_THRESHOLD}% on both GPUs."
  while true; do
    local c0 c1 f0 f1 u0 u1
    c0="$(gpu_pid_count "${PHYSICAL_GPU_A}")"
    c1="$(gpu_pid_count "${PHYSICAL_GPU_B}")"
    f0="$(gpu_free_mem "${PHYSICAL_GPU_A}")"
    f1="$(gpu_free_mem "${PHYSICAL_GPU_B}")"
    u0="$(gpu_util "${PHYSICAL_GPU_A}")"
    u1="$(gpu_util "${PHYSICAL_GPU_B}")"

    if [[ "${c0}" -eq 0 && "${c1}" -eq 0 ]]; then
      log "Target GPUs are idle. free_mem=[${f0},${f1}] MiB util=[${u0},${u1}]%. Starting jobs."
      break
    fi

    if [[ "${f0}" -ge "${FREE_MEM_THRESHOLD_MB}" && "${f1}" -ge "${FREE_MEM_THRESHOLD_MB}" \
       && "${u0}" -le "${UTIL_THRESHOLD}" && "${u1}" -le "${UTIL_THRESHOLD}" ]]; then
      log "Target GPUs are available by threshold. free_mem=[${f0},${f1}] MiB util=[${u0},${u1}]%. Starting jobs."
      break
    fi

    log "Still busy: gpu${PHYSICAL_GPU_A}(pid=${c0},free=${f0}MiB,util=${u0}%), gpu${PHYSICAL_GPU_B}(pid=${c1},free=${f1}MiB,util=${u1}%). Sleep ${WAIT_SECONDS}s."
    sleep "${WAIT_SECONDS}"
  done
}

run_seed_bg() {
  local physical_gpu="$1"
  local seed="$2"
  local seed_log="${RUN_DIR}/seed_${seed}.log"

  log "Launching seed=${seed} on physical GPU ${physical_gpu}. log=${seed_log}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY_BIN}" "${ROOT}/eval_retrieval_bm25.py" \
    --db-path "${ROOT}/patent.sqlite" \
    --table patents \
    --finetuned-model-path "${FINETUNED_MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_PREFIX}/seed_${seed}" \
    --query-year "${QUERY_YEAR}" \
    --window-years "${WINDOW_YEARS}" \
    --query-size "${QUERY_SIZE}" \
    --query-pool-multiplier "${QUERY_POOL_MULTIPLIER}" \
    --max-query-scan "${MAX_QUERY_SCAN}" \
    --query-source-citation-csv "${QUERY_SOURCE_CITATION_CSV}" \
    --hard-neg-per-query "${HARD_NEG_PER_QUERY}" \
    --ipc-semantic-per-query "${IPC_SEMANTIC_PER_QUERY}" \
    --ipc-random-per-query "${IPC_RANDOM_PER_QUERY}" \
    --cross-ipc-hard-per-query "${CROSS_IPC_HARD_PER_QUERY}" \
    --bm25-top-k "${BM25_TOP_K}" \
    --cross-ipc-bm25-top-k "${CROSS_IPC_BM25_TOP_K}" \
    --bm25-pool-per-ipc "${BM25_POOL_PER_IPC}" \
    --global-bm25-pool-size "${GLOBAL_BM25_POOL_SIZE}" \
    --exclude-pubno-file "${EXCLUDE_PUBNO_FILE}" \
    --device "${LOCAL_DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --rank-chunk-size "${RANK_CHUNK_SIZE}" \
    --sql-fetch-batch "${SQL_FETCH_BATCH}" \
    --seed "${seed}" \
    > "${seed_log}" 2>&1 &

  LAST_PID="$!"
}

aggregate_three_seeds() {
  log "Aggregating seeds ${SEEDS_CSV} to multiseed_summary.json and multiseed_metrics_table.csv"
  "${PY_BIN}" - "${RUN_DIR}" "${RUN_PREFIX}" "${SEEDS_CSV}" <<'PY'
import csv
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_prefix = sys.argv[2]
seeds = [int(x.strip()) for x in sys.argv[3].split(",") if x.strip()]
if not seeds:
    raise ValueError("empty seeds")

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


def mean_std(vals):
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(st.mean(vals)), "std": float(st.stdev(vals))}


runs = []
for seed in seeds:
    summary_path = run_dir / f"seed_{seed}" / "summary_topk_retrieval.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs.append(
        {
            "seed": seed,
            "run_name": f"{run_prefix}/seed_{seed}",
            "summary_path": str(summary_path),
            "metrics": summary["metrics"],
            "paired_comparison": summary.get("paired_comparison", {}),
        }
    )

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
    "runner": "/home/ljh/data1/patent/eval_retrieval_bm25.py",
    "runs": runs,
    "aggregate": agg,
}

out_json = run_dir / "multiseed_summary.json"
out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

out_csv = run_dir / "multiseed_metrics_table.csv"
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
PY
}

main() {
  parse_seeds "${SEEDS_CSV}"

  if [[ "${#SEEDS[@]}" -lt 3 ]]; then
    echo "Need at least 3 seeds in SEEDS_CSV, got ${SEEDS_CSV}" >&2
    exit 1
  fi

  log "Run config: query_size=${QUERY_SIZE}, hard_neg=${HARD_NEG_PER_QUERY}, ipc_semantic_neg=${IPC_SEMANTIC_PER_QUERY}, ipc_random_neg=${IPC_RANDOM_PER_QUERY}, cross_ipc_hard_neg=${CROSS_IPC_HARD_PER_QUERY}, bm25_top_k=${BM25_TOP_K}, cross_ipc_bm25_top_k=${CROSS_IPC_BM25_TOP_K}, bm25_pool_per_ipc=${BM25_POOL_PER_IPC}, global_bm25_pool_size=${GLOBAL_BM25_POOL_SIZE}, seeds=${SEEDS_CSV}"
  wait_for_gpu_idle

  local seed_a seed_b seed_c pid_a pid_b pid_c rc_a rc_b rc_c
  seed_a="${SEEDS[0]}"
  seed_b="${SEEDS[1]}"
  seed_c="${SEEDS[2]}"

  run_seed_bg "${PHYSICAL_GPU_A}" "${seed_a}"
  pid_a="${LAST_PID}"
  run_seed_bg "${PHYSICAL_GPU_B}" "${seed_b}"
  pid_b="${LAST_PID}"

  wait "${pid_a}" || rc_a=$?
  rc_a="${rc_a:-0}"
  wait "${pid_b}" || rc_b=$?
  rc_b="${rc_b:-0}"

  if [[ "${rc_a}" -ne 0 || "${rc_b}" -ne 0 ]]; then
    log "Parallel seeds failed: seed${seed_a}_rc=${rc_a}, seed${seed_b}_rc=${rc_b}."
    exit 1
  fi

  log "seed ${seed_a}/${seed_b} finished, launching seed ${seed_c} on GPU ${PHYSICAL_GPU_A}"
  run_seed_bg "${PHYSICAL_GPU_A}" "${seed_c}"
  pid_c="${LAST_PID}"
  wait "${pid_c}" || rc_c=$?
  rc_c="${rc_c:-0}"

  if [[ "${rc_c}" -ne 0 ]]; then
    log "seed ${seed_c} failed: rc=${rc_c}."
    exit 1
  fi

  aggregate_three_seeds
  log "DONE. Results saved under ${RUN_DIR}"
}

main "$@"
