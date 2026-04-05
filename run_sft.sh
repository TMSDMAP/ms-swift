#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 将 Hugging Face 缓存与临时文件统一放到数据盘，避免 / 和 /tmp 爆满
export HF_HOME="/home/ljh/data1/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
# ms-swift 的 dataset loader 会优先使用 ModelScope 的 cache_dir
export MODELSCOPE_CACHE="/home/ljh/data1/modelscope_cache"
export PACKING_CACHE="${MODELSCOPE_CACHE}/packing_cache"

export TMPDIR="/home/ljh/data1/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${MODELSCOPE_CACHE}" "${PACKING_CACHE}" "${TMPDIR}"

SWIFT_BIN="${SWIFT_BIN:-/data3/ljh/anaconda3/envs/ms-swift/bin/swift}"
if [[ ! -x "${SWIFT_BIN}" ]]; then
    SWIFT_BIN="$(command -v swift 2>/dev/null || true)"
fi
if [[ -z "${SWIFT_BIN}" ]]; then
    echo "ERROR: swift command not found. Set SWIFT_BIN or add swift to PATH." >&2
    exit 1
fi
# 环境变量设定 (针对 LLaVE 或特定的 InfoNCE 魔改扩展)
export INFONCE_TEMPERATURE=0.02     # 低温度系数，放大难负样本的梯度信号
export LLAVE_ALPHA=9.0              # LLaVE 的困难样本惩罚系数
export INFONCE_USE_BATCH=True       # 开启跨设备 In-batch 负样本
export INFONCE_HARD_NEGATIVES=4     # 修正：我们之前的数据脚本里固定挖了 4 个难负样本
export INFONCE_MASK_FAKE_NEGATIVE=True  # 屏蔽过于相似的 In-batch 假负样本，防止真·强负例坍塌
export INFONCE_FAKE_NEG_MARGIN=0.1      # 假负样本的掩码阈值余量
export INFONCE_INCLUDE_QQ=True          # 在分母中增加 query 间相似度作为负例
export INFONCE_INCLUDE_DD=True          # 在分母中增加 doc 间相似度作为负例

RESUME_FROM="${RESUME_FROM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Qwen3_10W_rerun}"

SWIFT_ARGS=(
    sft
    --model /home/ljh/data1/Qwen3-embedding-4B
    --task_type embedding
    --train_type full
    --dataset /home/ljh/data1/dataset/train.jsonl
    --val_dataset /home/ljh/data1/dataset/val.jsonl
    --attn_impl flash_attn
    --torch_dtype bfloat16
    --eval_strategy steps
    --output_dir "${OUTPUT_DIR}"
    --save_steps 100
    --eval_steps 100
    --save_total_limit 5
    --logging_steps 10
    --num_train_epochs 1
    --max_length 2048
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 8
    --gradient_accumulation_steps 2
    --gradient_checkpointing true
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --learning_rate 8e-6
    --warmup_ratio 0.08
    --loss_type infonce
    --temperature 0.02
    --seed 42
    --dataloader_drop_last true
    --deepspeed zero3
)

if [[ -n "${RESUME_FROM}" ]]; then
    echo "INFO: resume training from checkpoint: ${RESUME_FROM}"
    SWIFT_ARGS+=(--resume_from_checkpoint "${RESUME_FROM}")
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
else
    unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true
fi

HF_SKIP_TORCH_LOAD_IS_SAFE_CHECK=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
"${SWIFT_BIN}" "${SWIFT_ARGS[@]}"
