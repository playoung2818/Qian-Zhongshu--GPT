# /// script
# dependencies = [
#   "accelerate>=1.10,<2",
#   "bitsandbytes>=0.49,<1",
#   "datasets>=4,<5",
#   "huggingface-hub>=1.0,<2",
#   "peft>=0.19,<1",
#   "trackio>=0.12,<1",
#   "transformers>=4.57,<6",
#   "trl>=0.23,<1",
# ]
# ///
"""Fine-tune a private Lincoln-style Qwen LoRA adapter on Hugging Face Jobs."""
from __future__ import annotations

import os

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "Playoung2818/lincoln-style-chat-data"
MODEL_ID = "Playoung2818/lincoln-qwen2.5-7b-lora"
OUTPUT_DIR = "lincoln-qwen2.5-7b-lora"


def main() -> None:
    token = os.environ["HF_TOKEN"]
    dataset = load_dataset(DATASET_ID, token=token)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=token,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            max_length=1024,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=1,
            bf16=True,
            report_to="trackio",
            project="lincoln-style-model",
            run_name="qwen2.5-7b-lora-v1",
            push_to_hub=False,
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    api = HfApi(token=token)
    api.create_repo(MODEL_ID, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=MODEL_ID,
        repo_type="model",
        folder_path=OUTPUT_DIR,
        ignore_patterns=["checkpoint-*/*"],
        commit_message="Train Lincoln style Qwen2.5-7B LoRA adapter"
    )


if __name__ == "__main__":
    main()
