"""
Generate SQL predictions with Qwen2.5 + optional LoRA.

Loading matches train/qwen_lora_finetune: full-precision base (float32), tokenizer from base
(checkpoint tokenizers are unreliable). LoRA is loaded with config keys filtered for this peft version.
"""

from typing import Optional

import inspect
import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from tqdm import tqdm


def build_prompt(question: str, db_info: str) -> str:
    return (
        "### Complete sqlite SQL query only and with no explanation, "
        "and do not select extra columns that are not explicitly requested in the query.\n"
        "### Sqlite SQL tables, with their properties:\n"
        "#\n"
        f"{db_info.strip()}\n"
        "#\n"
        f"### {question}\n"
        "SELECT \n"
    )


def format_sql_response(text: str) -> str:
    text = text.strip().replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    for stop in ["###", "<|", "\x00"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    if not text.upper().startswith("SELECT"):
        text = "SELECT " + text
    text = text.rstrip(".,;")
    return text


def load_model(base_model: str, adapter_path: Optional[str] = None):
    """Same dtype rules as qwen_lora_finetune default (FP16 mixed train → float32 base weights)."""
    print(f"Loading tokenizer: {base_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except (AttributeError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, use_fast=False
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (float32): {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    model.config.use_cache = True

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
        allowed = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
        peft_config = LoraConfig(**{k: v for k, v in raw.items() if k in allowed})
        model = PeftModel.from_pretrained(model, adapter_path, config=peft_config)
    else:
        print("No adapter — base model only")

    model.eval()
    return model, tokenizer


def generate_sql_batch(model, tokenizer, prompts: list, max_new_tokens: int = 128) -> list:
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Same base as training (qwen_lora_finetune).",
    )
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--validation_data", type=str, default="data/validation_sql_clear.json")
    parser.add_argument("--output", type=str, default="predictions/qwen3b_sql.txt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading validation data: {args.validation_data}")
    with open(args.validation_data) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples")

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    dataset = dataset[args.resume_from:]
    mode = "a" if args.resume_from > 0 else "w"

    with open(args.output, mode) as f:
        for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
            batch = dataset[batch_start: batch_start + args.batch_size]
            prompts = [build_prompt(e["question"], e["db_info"]) for e in batch]
            try:
                raws = generate_sql_batch(model, tokenizer, prompts, args.max_new_tokens)
                for raw in raws:
                    sql = format_sql_response(raw)
                    f.write(sql + "\n")
            except Exception as e:
                print(f"\nError at batch {batch_start}: {e}")
                for _ in batch:
                    f.write("\n")
            f.flush()

    print(f"\nDone! Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
