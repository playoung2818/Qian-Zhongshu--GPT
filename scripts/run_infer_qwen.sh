#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_infer_qwen.sh

python inference_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --lora_path saved_models/qwen2_5_qian_lora \
  --instruction 用梁实秋风格的语言改写，保持原意： \
  --input_path test_data/test.txt \
  --output_path test_data/output_qwen.txt

