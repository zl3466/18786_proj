"""
Generate SQL predictions using Qwen + optional LoRA adapter.

Base load matches default train/qwen_lora_finetune (no --bf16): float32 weights,
same as fp16 mixed-precision training (fp32 base, autocast fp16 during train).
"""

import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def build_prompt(question: str, db_info: str, prompt_format: str = "alpaca") -> str:
    """prompt_format: alpaca | gpt | alpaca_supress"""
    if prompt_format == "alpaca":
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n\n"
            f"Convert text to sql: {question}\n"
            f"{db_info}\n\n"
            "### Response:\n\n"
        )
    if prompt_format == "alpaca_supress":
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request with no explanation.\n"
            "### Instruction:\n"
            f"Convert text to sql: {question}\n"
            f"{db_info}\n\n"
            "### Response:\n\n"
        )
    if prompt_format == "gpt":
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
    raise ValueError(f"Unknown prompt_format: {prompt_format!r} (use alpaca, gpt, or alpaca_supress)")


def format_sql_response(
    text: str,
    extract_skeleton: bool = False,
    prompt_format: str = "gpt",
) -> str:
    text = text.strip()
    # Training target: sql_skeleton | norm_sql — keep executable SQL only
    if extract_skeleton and "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if len(parts) > 1:
            text = parts[-1]
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    # Stop at common delimiters that signal end of SQL
    for stop in ["###", "<|", "\x00"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    # gpt format continues after "SELECT \n"; prepend SELECT if the model omitted it.
    # Alpaca formats complete after "### Response:" — leave the model text unchanged (no SELECT prepend).
    if prompt_format == "gpt" and not text.upper().startswith("SELECT"):
        text = "SELECT " + text
    text = text.rstrip(".,;")
    return text


def load_model(base_model: str, adapter_path: str = None):
    """Match train/qwen_lora_finetune.load_qwen_model with FP16=True, BF16=False (float32 base)."""
    print(f"Loading base model: {base_model}")
    # Do not load tokenizer from adapter_path: different checkpoints ship tokenizer.json
    # that breaks some transformers versions (e.g. extra_special_tokens as list → Fast tokenizer crash).
    # LoRA uses the base model vocabulary; always use base_model for the tokenizer.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
    except (AttributeError, TypeError, ValueError) as e:
        print(f"Fast tokenizer failed ({e}); retrying with use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32
    print(f"Base weight dtype: {torch_dtype} (matches default training: fp16 mixed, fp32 base)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )

    model.config.use_cache = True
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("No adapter — running base model only")
    model.eval()
    return model, tokenizer


def generate_sql_batch(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> list:
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

    # temperature == 0 → greedy (deterministic). top_p / nucleus sampling only apply when do_sample=True.
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

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
        help="Must match training base (see train/qwen_lora_finetune.py QwenLoraConfig.MODEL_NAME).",
    )
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--validation_data", type=str, default="data/validation_sql_clear.json")
    parser.add_argument("--output", type=str, default="predictions/llama3b_sql_chatgpt.txt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="alpaca",
        choices=["alpaca", "gpt", "alpaca_supress"],
        help="alpaca: Alpaca ### Instruction / ### Response. gpt: schema + SELECT continuation. "
        "alpaca_supress: Alpaca variant with extra column constraint and tighter newlines.",
    )
    parser.add_argument(
        "--extract_skeleton",
        action="store_true",
        help="If output looks like skeleton | sql, keep only the part after the last '|'.",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="If set, save all built prompts for this run as a JSON array of "
        "{index, prompt_format, db_id?, question?, prompt} (index is the row in the full validation file).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy decoding (deterministic). >0 enables sampling and uses --top_p.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling (0–1). Only used when --temperature > 0.",
    )
    args = parser.parse_args()

    if args.temperature <= 0 and args.top_p < 1.0:
        print(
            "Note: --top_p is ignored with greedy decoding (--temperature 0). "
            "Use e.g. --temperature 0.7 --top_p 0.9 for nucleus sampling."
        )

    print(f"Prompt format: {args.prompt_format}")

    print(f"Loading validation data: {args.validation_data}")
    with open(args.validation_data) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples")

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    dataset = dataset[args.resume_from:]
    mode = "a" if args.resume_from > 0 else "w"

    prompt_records = [] if args.prompts_json else None

    with open(args.output, mode) as f:
        for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
            batch = dataset[batch_start: batch_start + args.batch_size]
            prompts = [
                build_prompt(e["question"], e["db_info"], prompt_format=args.prompt_format)
                for e in batch
            ]
            if prompt_records is not None:
                for i, e in enumerate(batch):
                    idx = args.resume_from + batch_start + i
                    rec = {"index": idx, "prompt": prompts[i], "prompt_format": args.prompt_format}
                    if e.get("db_id") is not None:
                        rec["db_id"] = e["db_id"]
                    if e.get("question") is not None:
                        rec["question"] = e["question"]
                    prompt_records.append(rec)
            try:
                raws = generate_sql_batch(
                    model,
                    tokenizer,
                    prompts,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for raw in raws:
                    sql = format_sql_response(
                        raw,
                        extract_skeleton=args.extract_skeleton,
                        prompt_format=args.prompt_format,
                    )
                    f.write(sql + "\n")
            except Exception as e:
                print(f"\nError at batch {batch_start}: {e}")
                for _ in batch:
                    f.write("\n")
            f.flush()

    if args.prompts_json:
        pj = args.prompts_json
        pj_dir = os.path.dirname(pj)
        if pj_dir:
            os.makedirs(pj_dir, exist_ok=True)
        with open(pj, "w", encoding="utf-8") as pf:
            json.dump(prompt_records, pf, indent=2, ensure_ascii=False)
        print(f"Built prompts saved to: {pj}")

    print(f"\nDone! Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()