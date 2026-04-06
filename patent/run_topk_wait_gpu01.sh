#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljh/data1/patent"
PY_BIN="/data3/ljh/anaconda3/envs/ms-swift/bin/python"
RUN_PREFIX="${RUN_PREFIX:-bm25_multiseed_formal_fast_$(date +%Y%m%d)_q60_s3_gpu01_wait}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-50000}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-20}"

OUTPUT_DIR="${ROOT}/eval_outputs"
RUN_DIR="${OUTPUT_DIR}/${RUN_PREFIX}"
mkdir -p "${RUN_DIR}"
LAUNCHER_LOG="${RUN_DIR}/launcher.log"

log() {
  local msg="$*"
  echo "[$(date '+%F %T')] ${msg}" | tee -a "${LAUNCHER_LOG}" >&2
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
  log "Waiting for physical GPU 0 and 1 to become idle/available..."
  log "Availability thresholds: free_mem>=${FREE_MEM_THRESHOLD_MB}MiB, util<=${UTIL_THRESHOLD}% on both GPUs."
  while true; do
    local c0 c1 f0 f1 u0 u1
    c0="$(gpu_pid_count 0)"
    c1="$(gpu_pid_count 1)"
    f0="$(gpu_free_mem 0)"
    f1="$(gpu_free_mem 1)"
    u0="$(gpu_util 0)"
    u1="$(gpu_util 1)"

    if [[ "${c0}" -eq 0 && "${c1}" -eq 0 ]]; then
      log "GPU 0/1 are idle. free_mem=[${f0},${f1}] MiB util=[${u0},${u1}]%. Starting jobs."
      break
    fi

    if [[ "${f0}" -ge "${FREE_MEM_THRESHOLD_MB}" && "${f1}" -ge "${FREE_MEM_THRESHOLD_MB}" \
       && "${u0}" -le "${UTIL_THRESHOLD}" && "${u1}" -le "${UTIL_THRESHOLD}" ]]; then
      log "GPU 0/1 are available by threshold. free_mem=[${f0},${f1}] MiB util=[${u0},${u1}]%. Starting jobs."
      break
    fi

    log "Still busy: gpu0(pid=${c0},free=${f0}MiB,util=${u0}%), gpu1(pid=${c1},free=${f1}MiB,util=${u1}%). Sleep ${WAIT_SECONDS}s."
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
    --finetuned-model-path "/home/ljh/data1/output_backup/Qwen3_10W/v14-20260331-153258_independent_20260401-190135/checkpoint-700" \
    --base-model-path "/home/ljh/data1/Qwen3-embedding-4B" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_PREFIX}/seed_${seed}" \
    --query-year 2021 \
    --window-years 18 \
    --query-size 60 \
    --query-pool-multiplier 25 \
    --max-query-scan 300000 \
    --query-source-citation-csv "${ROOT}/eval_outputs/benchmark_20260405-105001/citation_quartet_scores.csv" \
    --hard-neg-per-query 8 \
    --ipc-random-per-query 8 \
    --bm25-top-k 80 \
    --bm25-pool-per-ipc 300 \
    --exclude-pubno-file "${ROOT}/eval_outputs/exclude_pubnos_trainval_full_20260405.txt" \
    --device cuda:0 \
    --batch-size 64 \
    --max-length 768 \
    --rank-chunk-size 64 \
    --sql-fetch-batch 16384 \
    --seed "${seed}" \
    > "${seed_log}" 2>&1 &

  LAST_PID="$!"
}

aggregate_three_seeds() {
  log "Aggregating seeds [42,43,44] to multiseed_summary.json and multiseed_metrics_table.csv"
  "${PY_BIN}" - "${RUN_DIR}" "${RUN_PREFIX}" <<'PY'
import csv
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_prefix = sys.argv[2]
seeds = [42, 43, 44]

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
  wait_for_gpu_idle

  local pid42 pid43 pid44 rc42 rc43 rc44
  run_seed_bg 0 42
  pid42="${LAST_PID}"
  run_seed_bg 1 43
  pid43="${LAST_PID}"

  wait "${pid42}" || rc42=$?
  rc42="${rc42:-0}"
  wait "${pid43}" || rc43=$?
  rc43="${rc43:-0}"

  if [[ "${rc42}" -ne 0 || "${rc43}" -ne 0 ]]; then
    log "Parallel seeds failed: seed42_rc=${rc42}, seed43_rc=${rc43}."
    exit 1
  fi

  log "seed 42/43 finished, launching seed 44 on GPU 0"
  run_seed_bg 0 44
  pid44="${LAST_PID}"
  wait "${pid44}" || rc44=$?
  rc44="${rc44:-0}"

  if [[ "${rc44}" -ne 0 ]]; then
    log "seed 44 failed: rc=${rc44}."
    exit 1
  fi

  aggregate_three_seeds
  log "DONE. Results saved under ${RUN_DIR}"
}

main "$@"
