#!/bin/bash

# 环境变量设定 (针对 LLaVE 或特定的 InfoNCE 魔改扩展)
export INFONCE_TEMPERATURE=0.02     # 低温度系数，放大难负样本的梯度信号
export LLAVE_ALPHA=9.0              # LLaVE 的困难样本惩罚系数
export INFONCE_USE_BATCH=True       # 开启跨设备 In-batch 负样本 (重要！)
export INFONCE_HARD_NEGATIVES=4     # 修正：我们之前的数据脚本里固定挖了 4 个难负样本

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model /home/ljh/data1/Qwen3-embedding-4B \
    --task_type embedding \
    --train_type full \
    --dataset /home/ljh/data1/dataset/train.jsonl \
    --val_dataset /home/ljh/data1/dataset/val.jsonl \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --eval_strategy steps \
    --output_dir output/Qwen3_10W \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --query_max_length 512 \
    --max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --loss_type infonce \
    --temperature 0.02 \
    --dataloader_drop_last true \
    --deepspeed zero3