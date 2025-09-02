# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import random
import shutil
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# --- 路径 & 配置 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

BASE_MODEL_PATH = "./models/gemma-3-270m"
PROMPT_TRAIN_CSV = "./data/train.csv"
PROMPT_VAL_CSV = "./data/test.csv"
TOKENIZED_TRAIN_PATH = "./cached/tokenized_train_gen"
TOKENIZED_VAL_PATH = "./cached/tokenized_val_gen"
OUTPUT_DIR = "./checkpoints/lora_gemma_generation"

# --- Lora Config ---
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1

# --- Training Args ---
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-4

# --- Tokenizer & 模型 ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
)

# --- LoRA 配置 & 注入 ---
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# --- 预处理函数 ---
def preprocess_data(examples, max_len=512):
    batch = {k: [] for k in ["input_ids", "attention_mask", "labels"]}
    
    for text, label_str in zip(examples["text"], examples["label"]):
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]

        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
            labels = [-100] * pad_len + labels
        
        attention_mask = [0] * pad_len + [1] * (max_len - pad_len)

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)
        
    return batch

# --- 数据集加载与缓存 ---
PREPROC_SIGNATURE = {
    "max_len": 512,
    "truncation_side": "left",
    "prompt_format": "text_label_concat",
}

def _sig_hex(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

PREPROC_SIG_HEX = _sig_hex(PREPROC_SIGNATURE)

def build_or_load_datasets():
    sig_file = os.path.join(os.path.dirname(TOKENIZED_TRAIN_PATH), f"sig_{PREPROC_SIG_HEX}.json")

    if os.path.exists(TOKENIZED_TRAIN_PATH) and os.path.exists(sig_file):
        print(f"✅ 缓存签名 {PREPROC_SIG_HEX} 一致，加载 tokenized 数据集...")
        return load_from_disk(TOKENIZED_TRAIN_PATH), load_from_disk(TOKENIZED_VAL_PATH)
    
    print("🔄 重建 tokenize 数据...")
    if os.path.exists(TOKENIZED_TRAIN_PATH):
        shutil.rmtree(TOKENIZED_TRAIN_PATH)
    if os.path.exists(TOKENIZED_VAL_PATH):
        shutil.rmtree(TOKENIZED_VAL_PATH)

    train_df = pd.read_csv(PROMPT_TRAIN_CSV)
    val_df = pd.read_csv(PROMPT_VAL_CSV)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(preprocess_data, batched=True, remove_columns=["text", "label"])
    val_ds = val_ds.map(preprocess_data, batched=True, remove_columns=["text", "label"])

    train_ds.save_to_disk(TOKENIZED_TRAIN_PATH)
    val_ds.save_to_disk(TOKENIZED_VAL_PATH)

    with open(sig_file, "w", encoding="utf-8") as f:
        json.dump(PREPROC_SIGNATURE, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存新 tokenized 数据及签名 {PREPROC_SIG_HEX}。")
    return train_ds, val_ds

# --- 训练 ---
if __name__ == "__main__":
    random.seed(42)
    train_dataset, val_dataset = build_or_load_datasets()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=20,
        save_strategy="epoch",
        # evaluation_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        # load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print("\n" + "=" * 30)
    print("🚀 开始 LoRA 微调训练...")
    print("=" * 30)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 训练完成，最终模型已保存至: {OUTPUT_DIR}")