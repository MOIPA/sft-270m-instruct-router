# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

# --- 路径配置 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

BASE_MODEL_PATH = "./models/gemma-3-270m-it"
LORA_MODEL_PATH = "./checkpoints/lora_gemma_generation"
MERGED_MODEL_PATH = "./models/merged_gemma_lora"

# --- 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_lora_with_base_model():
    """
    加载基础模型和LoRA适配器，将它们合并，并保存合并后的模型。
    """
    print("=" * 30)
    print("🚀 开始融合 LoRA 模型与基础模型...")
    print("=" * 30)

    # 1. 加载分词器
    print(f"🔄 正在从 '{BASE_MODEL_PATH}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载完成。")

    # 2. 加载基础模型
    print(f"🔄 正在从 '{BASE_MODEL_PATH}' 加载基础模型...")
    offload_folder = "./offload"
    os.makedirs(offload_folder, exist_ok=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map='auto', # 自动选择设备
        offload_folder=offload_folder
    )
    print("✅ 基础模型加载完成。")

    # 3. 加载LoRA适配器并与基础模型合并
    print(f"🔄 正在从 '{LORA_MODEL_PATH}' 加载LoRA适配器并进行合并...")
    # 加载PeftModel
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, offload_folder=offload_folder)
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    print("✅ LoRA模型合并完成。")

    # 4. 保存合并后的模型和分词器
    print(f"💾 正在将合并后的模型保存到 '{MERGED_MODEL_PATH}'...")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)

    print("🔄 正在将模型移至CPU以便保存 (这可能需要一些时间和内存)...")
    merged_model = merged_model.to("cpu")
    print("✅ 模型已移至CPU。")

    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    # --- 5. 复制 tokenizer.model 以确保兼容性 ---
    print("🔄 正在复制 tokenizer.model 文件以确保GGUF转换兼容性...")
    try:
        import shutil
        source_tokenizer_model = os.path.join(BASE_MODEL_PATH, "tokenizer.model")
        dest_tokenizer_model = os.path.join(MERGED_MODEL_PATH, "tokenizer.model")
        if os.path.exists(source_tokenizer_model):
            shutil.copyfile(source_tokenizer_model, dest_tokenizer_model)
            print("✅ tokenizer.model 复制完成。")
        else:
            print("⚠️ 警告: 在源目录中未找到 tokenizer.model，跳过复制。")
    except Exception as e:
        print(f"❌ 复制 tokenizer.model 时出错: {e}")

    print(f"🎉 成功！合并后的模型已保存至: {MERGED_MODEL_PATH}")
    print("=" * 30)


if __name__ == "__main__":
    merge_lora_with_base_model()
