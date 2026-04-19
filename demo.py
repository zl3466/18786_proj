#!/usr/bin/env python3
"""Interactive chat with Qwen2.5-3B-Instruct, optionally with a LoRA adapter."""

import argparse
import inspect
import json
import os
import sys

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from qwen_lora_finetune_mine import QwenLoraConfig


def parse_args():
    p = argparse.ArgumentParser(description="Chat with base Qwen or a LoRA checkpoint")
    p.add_argument(
        "--model_name",
        type=str,
        default=QwenLoraConfig.MODEL_NAME,
        help="Hugging Face model id or local path (base weights)",
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Directory with LoRA adapter (e.g. ./qwen3b-lora-adapter/checkpoint-500). Omit for base model only.",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Load in bfloat16 (recommended on supported GPUs)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens per reply",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    return p.parse_args()


def pick_dtype(use_bf16: bool):
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def input_device_for(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return next(model.parameters()).device


def load_lora_adapter(model: torch.nn.Module, adapter_path: str) -> PeftModel:
    """Load PEFT weights; filter adapter_config keys so older peft can load configs from newer trainers."""
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    allowed = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    dropped = set(raw) - set(filtered)
    if dropped:
        print(
            "Note: ignoring adapter_config keys not supported by this peft version "
            f"(upgrade peft to preserve them): {sorted(dropped)}"
        )
    peft_config = LoraConfig(**filtered)
    return PeftModel.from_pretrained(model, adapter_path, config=peft_config)


def main():
    args = parse_args()
    torch_dtype = pick_dtype(args.bf16)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=QwenLoraConfig.TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=QwenLoraConfig.TRUST_REMOTE_CODE,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = True

    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = load_lora_adapter(model, args.adapter_path)

    model.eval()
    dev = input_device_for(model)
    print("Ready. Type a message and press Enter. Empty line, quit, or exit to stop.\n")

    messages = []

    while True:
        try:
            line = input("============\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": line})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-6),
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\nAssistant: {reply}\n")

        messages.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    sys.exit(main())
