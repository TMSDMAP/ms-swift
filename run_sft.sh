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
    --output_dir output/Qwen3_10W_316 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero3