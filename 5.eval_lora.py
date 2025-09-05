
import os
import re
import json
import argparse
import pandas as pd
import torch
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 路径配置 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

BASE_MODEL_PATH = "./models/gemma-3-270m-it"
LORA_MODEL_PATH = "./checkpoints/lora_gemma_generation" # 默认LoRA路径
PROMPT_VAL_CSV = "./data/test.csv"
RESULTS_DIR = "./results/"

# --- 全局变量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = os.path.join(RESULTS_DIR, "lora_evaluation_results.json")
DETAILED_RESULTS_FILE = os.path.join(RESULTS_DIR, "lora_detailed_evaluation_results.csv")

# ---

def extract_json_output(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def calculate_argument_f1(predicted_args, true_args):
    if not isinstance(predicted_args, dict):
        predicted_args = {}
    if not isinstance(true_args, dict):
        true_args = {}

    predicted_set = set(predicted_args.items())
    true_set = set(true_args.items())

    tp = len(predicted_set.intersection(true_set))
    fp = len(predicted_set.difference(true_set))
    fn = len(true_set.difference(predicted_set))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_model(model, tokenizer, val_file, num_samples=None):
    df = pd.read_csv(val_file)
    if num_samples:
        df = df.head(num_samples)
    
    exact_match_count = 0
    tool_name_match_count = 0
    total_arg_f1 = 0
    total_arg_precision = 0
    total_arg_recall = 0
    total_count = 0
    results_data = []

    model.eval()                

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating LoRA model"):
        prompt = row["text"]
        true_label_str = row["label"]
        
        try:
            true_json_str = true_label_str.split('：', 1)[1]
            true_json = json.loads(true_json_str)
        except (json.JSONDecodeError, IndexError):
            continue

        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                top_p=None,
                top_k=None
            )
        
        generated_text = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        
        predicted_json = extract_json_output(generated_text)

        is_exact_match = False
        is_tool_name_match = False
        arg_precision, arg_recall, arg_f1 = 0, 0, 0

        if predicted_json:
            if predicted_json == true_json:
                exact_match_count += 1
                is_exact_match = True
            
            if predicted_json.get("tool_name") == true_json.get("tool_name"):
                tool_name_match_count += 1
                is_tool_name_match = True

            arg_precision, arg_recall, arg_f1 = calculate_argument_f1(
                predicted_json.get("arguments", {}),
                true_json.get("arguments", {})
            )
            total_arg_precision += arg_precision
            total_arg_recall += arg_recall
            total_arg_f1 += arg_f1

        total_count += 1
        results_data.append({
            'prompt': prompt,
            'ground_truth': true_json,
            'generated_text': generated_text,
            'predicted_json': predicted_json,
            'exact_match': is_exact_match,
            'tool_name_match': is_tool_name_match,
            'arg_precision': arg_precision,
            'arg_recall': arg_recall,
            'arg_f1': arg_f1
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(DETAILED_RESULTS_FILE, index=False)
    print(f"\n✅ Detailed evaluation results saved to {DETAILED_RESULTS_FILE}")

    exact_match_rate = exact_match_count / total_count if total_count > 0 else 0
    tool_name_accuracy = tool_name_match_count / total_count if total_count > 0 else 0
    avg_arg_precision = total_arg_precision / total_count if total_count > 0 else 0
    avg_arg_recall = total_arg_recall / total_count if total_count > 0 else 0
    avg_arg_f1 = total_arg_f1 / total_count if total_count > 0 else 0

    summary = {
        "exact_match_rate": exact_match_rate,
        "tool_name_accuracy": tool_name_accuracy,
        "average_argument_precision": avg_arg_precision,
        "average_argument_recall": avg_arg_recall,
        "average_argument_f1": avg_arg_f1,
        "total_samples": total_count,
    }
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA model for tool calling.")
    parser.add_argument("--lora_path", type=str, default=LORA_MODEL_PATH, help="Path to the LoRA adapter checkpoint.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate on. Defaults to all.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ).to(device)

    print(f"Loading LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path).to(device)
    model.eval() # Set the merged model to evaluation mode

    print("\n" + "=" * 30)
    print(f"1. Starting evaluation of LoRA model on {args.num_samples or 'all'} samples...")
    print("=" * 30)

    evaluation_summary = evaluate_model(model, tokenizer, PROMPT_VAL_CSV, num_samples=args.num_samples)

    print("\n--- LoRA Model Evaluation Summary ---")
    print(json.dumps(evaluation_summary, indent=2))

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\n✅ LoRA evaluation summary saved to {RESULTS_FILE}")
    print("\n" + "=" * 30)
    print(f"🎉 LoRA evaluation complete!")
    print("=" * 30)
