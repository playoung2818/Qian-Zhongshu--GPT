#!/usr/bin/env python3
"""QLoRA training script for Qwen2.5 style transfer."""
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class Row:
    instruction: str
    input: str
    output: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_file", required=True, help="jsonl with instruction/input/output")
    p.add_argument("--eval_file", default=None, help="optional jsonl with same schema")
    p.add_argument("--output_dir", default="saved_models/qwen2_5_lora")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def build_text(instruction: str, inp: str, out: str) -> str:
    return (
        "### 指令:\n"
        f"{instruction}\n\n"
        "### 输入:\n"
        f"{inp}\n\n"
        "### 输出:\n"
        f"{out}"
    )


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    ds = load_dataset("json", data_files=data_files)

    def preprocess(batch):
        texts = [
            build_text(inst, inp, out)
            for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        tok = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return tok

    remove_cols = ds["train"].column_names
    tokenized = ds.map(preprocess, batched=True, remove_columns=remove_cols)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"] if "validation" in tokenized else None

    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        **{strategy_arg: "steps" if eval_dataset is not None else "no"},
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
