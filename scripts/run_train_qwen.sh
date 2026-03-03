#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_train_qwen.sh

python train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file example_data/qianzhongshu_train.jsonl \
  --eval_file example_data/qianzhongshu_eval.jsonl \
  --output_dir saved_models/qwen2_5_qian_lora \
  --max_length 1024 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_steps 200 \
  --logging_steps 20

