#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljh/data1/patent"
PY_BIN="/data3/ljh/anaconda3/envs/ms-swift/bin/python"

RUN_PREFIX="${RUN_PREFIX:-bm25_multiseed_hardfair_$(date +%Y%m%d-%H%M%S)_q200_n32_gpu4567}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44,45}"

WAIT_SECONDS="${WAIT_SECONDS:-60}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-50000}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-20}"

QUERY_YEAR="${QUERY_YEAR:-2021}"
WINDOW_YEARS="${WINDOW_YEARS:-18}"
QUERY_SIZE="${QUERY_SIZE:-200}"
QUERY_POOL_MULTIPLIER="${QUERY_POOL_MULTIPLIER:-25}"
MAX_QUERY_SCAN="${MAX_QUERY_SCAN:-300000}"

HARD_NEG_PER_QUERY="${HARD_NEG_PER_QUERY:-32}"
IPC_SEMANTIC_PER_QUERY="${IPC_SEMANTIC_PER_QUERY:-16}"
IPC_RANDOM_PER_QUERY="${IPC_RANDOM_PER_QUERY:-16}"
CROSS_IPC_HARD_PER_QUERY="${CROSS_IPC_HARD_PER_QUERY:-16}"
BM25_TOP_K="${BM25_TOP_K:-300}"
CROSS_IPC_BM25_TOP_K="${CROSS_IPC_BM25_TOP_K:-3000}"
BM25_POOL_PER_IPC="${BM25_POOL_PER_IPC:-1200}"
GLOBAL_BM25_POOL_SIZE="${GLOBAL_BM25_POOL_SIZE:-200000}"

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

parse_int_csv() {
  local raw="$1"
  local out_name="$2"
  IFS=',' read -r -a _vals <<< "${raw}"
  if [[ "${#_vals[@]}" -eq 0 ]]; then
    echo "Empty CSV: ${raw}" >&2
    exit 1
  fi
  for v in "${_vals[@]}"; do
    if ! [[ "${v}" =~ ^[0-9]+$ ]]; then
      echo "Invalid integer value: ${v}" >&2
      exit 1
    fi
  done
  eval "${out_name}=(\"\${_vals[@]}\")"
}

gpu_pid_count() {
  local gpu_idx="$1"
  nvidia-smi -i "${gpu_idx}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk '/^[0-9]+$/ {c++} END {print c+0}'
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

wait_for_gpus() {
  log "Waiting for GPUs [${GPU_LIST}] to become idle/available..."
  log "Thresholds: free_mem>=${FREE_MEM_THRESHOLD_MB}MiB and util<=${UTIL_THRESHOLD}%"

  while true; do
    local all_idle=1
    local all_available=1
    local status=""

    for g in "${GPUS[@]}"; do
      local c f u
      c="$(gpu_pid_count "${g}")"
      f="$(gpu_free_mem "${g}")"
      u="$(gpu_util "${g}")"
      status+=" gpu${g}(pid=${c},free=${f}MiB,util=${u}%)"

      if [[ "${c}" -ne 0 ]]; then
        all_idle=0
      fi
      if [[ "${f}" -lt "${FREE_MEM_THRESHOLD_MB}" || "${u}" -gt "${UTIL_THRESHOLD}" ]]; then
        all_available=0
      fi
    done

    if [[ "${all_idle}" -eq 1 || "${all_available}" -eq 1 ]]; then
      log "GPU ready:${status}"
      break
    fi

    log "GPU busy:${status}. Sleep ${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
  done
}

run_seed_bg() {
  local physical_gpu="$1"
  local seed="$2"
  local seed_log="${RUN_DIR}/seed_${seed}.log"

  log "Launching seed=${seed} on GPU ${physical_gpu}. log=${seed_log}"
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

aggregate_multiseed() {
  log "Aggregating seeds ${SEEDS_CSV}"
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

agg = {"finetuned": {}, "base": {}, "delta_ft_minus_base": {}}
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
  parse_int_csv "${GPU_LIST}" GPUS
  parse_int_csv "${SEEDS_CSV}" SEEDS

  if [[ "${#SEEDS[@]}" -gt "${#GPUS[@]}" ]]; then
    echo "Need at least as many GPUs as seeds for this launcher. GPUs=${GPU_LIST}, seeds=${SEEDS_CSV}" >&2
    exit 1
  fi

  log "Run config: query_size=${QUERY_SIZE}, hard_neg=${HARD_NEG_PER_QUERY}, ipc_semantic=${IPC_SEMANTIC_PER_QUERY}, ipc_random=${IPC_RANDOM_PER_QUERY}, cross_ipc_hard=${CROSS_IPC_HARD_PER_QUERY}, bm25_top_k=${BM25_TOP_K}, cross_ipc_bm25_top_k=${CROSS_IPC_BM25_TOP_K}, bm25_pool_per_ipc=${BM25_POOL_PER_IPC}, global_bm25_pool_size=${GLOBAL_BM25_POOL_SIZE}, seeds=${SEEDS_CSV}, gpus=${GPU_LIST}"

  wait_for_gpus

  local -a pids=()
  local -a run_seeds=()
  local -a run_gpus=()

  local i
  for ((i=0; i<${#SEEDS[@]}; i++)); do
    run_seed_bg "${GPUS[$i]}" "${SEEDS[$i]}"
    pids+=("${LAST_PID}")
    run_seeds+=("${SEEDS[$i]}")
    run_gpus+=("${GPUS[$i]}")
  done

  local failed=0
  for ((i=0; i<${#pids[@]}; i++)); do
    local rc=0
    wait "${pids[$i]}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      log "seed ${run_seeds[$i]} on GPU ${run_gpus[$i]} failed: rc=${rc}"
      failed=1
    else
      log "seed ${run_seeds[$i]} on GPU ${run_gpus[$i]} finished"
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    log "At least one seed failed. Skip aggregation."
    exit 1
  fi

  aggregate_multiseed
  log "DONE. Results saved under ${RUN_DIR}"
}

main "$@"
