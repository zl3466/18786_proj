"""
Generate SQL predictions using Llama3-7B model on Spider validation dataset

This script loads Llama3-7B from Hugging Face, generates SQL queries
for the validation dataset, and saves predictions for evaluation.
"""

import json
import argparse
import dataclasses
import inspect
import os
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from tqdm import tqdm
import re


def _lora_config_from_adapter_dir(adapter_path: str) -> LoraConfig:
    """
    Load adapter_config.json but only pass fields supported by this installed peft version.
    Newer trainers (e.g. recent LLaMA-Factory) may save keys like alora_invocation_tokens that
    older LoraConfig rejects.
    """
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        valid = {f.name for f in dataclasses.fields(LoraConfig)}
    except TypeError:
        valid = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in raw.items() if k in valid}
    dropped = set(raw) - set(filtered)
    if dropped:
        print(
            f"   Note: ignoring adapter_config keys not in this peft's LoraConfig: "
            f"{sorted(dropped)}"
        )
    return LoraConfig(**filtered)

def format_sql_response(response_text: str, extract_skeleton: bool = False) -> str:
    """
    Format and clean SQL response from model.
    
    Args:
        response_text: Raw model response
        extract_skeleton: Whether to extract SQL from skeleton format
        
    Returns:
        Cleaned SQL query
    """
    response_text = response_text.strip()
    
    # Extract SQL from skeleton format if present
    if extract_skeleton and '|' in response_text:
        # Split on | and take the part after it (actual SQL)
        parts = response_text.split('|')
        if len(parts) > 1:
            response_text = parts[-1].strip()
    
    # Remove extra whitespace and newlines
    response_text = response_text.replace("\n", " ").replace("\t", " ")
    response_text = ' '.join(response_text.split())
    
    # Ensure it starts with SELECT
    if not response_text.upper().startswith('SELECT'):
        # Try to find SELECT in the text
        select_match = re.search(r'select\s+', response_text, re.IGNORECASE)
        if select_match:
            response_text = response_text[select_match.start():]
        else:
            response_text = 'SELECT ' + response_text
    
    # Remove trailing punctuation that might not be SQL
    response_text = response_text.rstrip('.,;')
    
    return response_text

def build_prompt(question: str, db_info: str, use_instruction_format: bool = True) -> str:
    """
    Build prompt for SQL generation.
    
    Args:
        question: Natural language question
        db_info: Database schema information
        use_instruction_format: Whether to use instruction-response format
        
    Returns:
        Formatted prompt
    """
    if use_instruction_format:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Convert text to sql: {question}
{db_info}

### Response:

"""
    else:
        # Alternative format similar to ChatGPT approach
        prompt = f"""### Complete sqlite SQL query only and with no explanation, and do not select extra columns that are not explicitly requested in the query.
### Sqlite SQL tables, with their properties:
#
{db_info}
#
### {question}
SELECT
"""
    return prompt

def generate_sql_batch(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
    stop_sequences: list = None,
) -> list:
    """
    Generate SQL for multiple prompts in one forward pass (left-padded for decoder-only LMs).

    Larger batch sizes improve GPU utilization when VRAM allows.
    """
    if not prompts:
        return []
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    results = []
    for j in range(len(prompts)):
        input_len = int(input_lens[j])
        gen_ids = outputs[j, input_len:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0].strip()
        results.append(generated_text)
    return results


def generate_sql(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
    stop_sequences: list = None,
) -> str:
    """
    Generate SQL query using the model.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        stop_sequences: Sequences that stop generation
        
    Returns:
        Generated SQL query
    """
    return generate_sql_batch(
        model,
        tokenizer,
        [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        stop_sequences=stop_sequences,
    )[0]

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    adapter_path: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    compile_model: bool = False,
):
    """
    Load causal LM and tokenizer. If adapter_path is set, loads a PEFT LoRA adapter on top of model_name.

    Args:
        model_name: HuggingFace model id or local path to **base** weights (also used for tokenizer when adapter_path is set)
        device: Device to load on ("auto", "cuda", "cpu")
        load_in_8bit: Whether to use 8-bit quantization
        adapter_path: Optional directory with LoRA adapter (e.g. ./qwen3b-lora-adapter/checkpoint-3000)

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer (always from base / merged checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if attn_implementation is None:
        attn_implementation = "sdpa" if torch.cuda.is_available() else "eager"

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": attn_implementation,
    }
    
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        lora_config = _lora_config_from_adapter_dir(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, config=lora_config)
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()

    if compile_model and hasattr(torch, "compile") and device == "cuda":
        print("Compiling model with torch.compile (first batches may be slow)...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    print(f"✅ Model loaded on {device}")
    print(f"   Model dtype: {model.dtype}")
    print(f"   Attention: {attn_implementation}")
    
    return model, tokenizer

def process_dataset(
    model,
    tokenizer,
    dataset: list,
    output_file: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    use_instruction_format: bool = True,
    extract_skeleton: bool = False,
    resume_from: int = 0,
    batch_size: int = 1,
):
    """
    Process validation dataset and generate predictions.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        dataset: List of validation entries
        output_file: Path to save predictions
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        use_instruction_format: Use instruction format or ChatGPT format
        extract_skeleton: Extract SQL from skeleton format
        resume_from: Index to resume from (for resuming interrupted runs)
        batch_size: Number of examples per generate() call (increase if VRAM allows)
    """
    # Check if we're resuming
    existing_predictions = []
    if resume_from > 0 and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_predictions = [line.strip() for line in f]
        print(f"Resuming from index {resume_from}, found {len(existing_predictions)} existing predictions")
    
    # Open output file in append mode if resuming
    mode = 'a' if resume_from > 0 else 'w'
    
    subset = dataset[resume_from:]
    stop_sequences = ["###", "\n\n\n"]
    do_sample = temperature > 0

    with open(output_file, mode) as f:
        if batch_size < 1:
            batch_size = 1

        pbar = tqdm(total=len(subset), desc="Generating predictions", initial=0)
        batch_start = 0
        while batch_start < len(subset):
            chunk = subset[batch_start : batch_start + batch_size]
            prompts = [
                build_prompt(
                    e["question"],
                    e["db_info"],
                    use_instruction_format=use_instruction_format,
                )
                for e in chunk
            ]
            global_idx = resume_from + batch_start
            try:
                raw_responses = generate_sql_batch(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    stop_sequences=stop_sequences,
                )
            except Exception as batch_err:
                print(f"\n⚠️ Batch generate failed at index {global_idx} ({batch_err}); retrying one-by-one.")
                raw_responses = []
                for p in prompts:
                    try:
                        raw_responses.append(
                            generate_sql(
                                model,
                                tokenizer,
                                p,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=do_sample,
                                stop_sequences=stop_sequences,
                            )
                        )
                    except Exception as e:
                        print(f"\n❌ Error processing: {e}")
                        raw_responses.append("")

            for raw_response in raw_responses:
                try:
                    sql = format_sql_response(raw_response, extract_skeleton=extract_skeleton)
                    f.write(sql + "\n")
                except Exception as e:
                    print(f"\n❌ Error formatting response: {e}")
                    f.write("\n")
                f.flush()

            pbar.update(len(chunk))
            batch_start += len(chunk)

        pbar.close()
    
    print(f"\n✅ Predictions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL predictions using Llama3-7B on Spider validation dataset"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='HuggingFace model id or local path to base/merged weights (tokenizer loads from here too)',
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Optional directory with a LoRA adapter (PEFT). Base weights come from --model_name.',
    )
    parser.add_argument(
        '--validation_data',
        type=str,
        default='data/validation_sql_clear.json',
        help='Path to validation dataset JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='eval/predictions/llama3_7b_predictions.txt',
        help='Output file path for predictions'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0.0 for greedy decoding)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling parameter'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        help='Load model in 8-bit quantization (saves memory)'
    )
    parser.add_argument(
        '--use_chatgpt_format',
        action='store_true',
        help='Use ChatGPT-style prompt format instead of instruction format'
    )
    parser.add_argument(
        '--extract_skeleton',
        action='store_true',
        help='Extract SQL from skeleton format (if model outputs skeleton | sql)'
    )
    parser.add_argument(
        '--resume_from',
        type=int,
        default=0,
        help='Resume from this index (for resuming interrupted runs)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Examples per generate() call. Increase (e.g. 8–16) if VRAM headroom; reduces GPU idle time.',
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default='auto',
        choices=['auto', 'eager', 'sdpa', 'flash_attention_2'],
        help='Attention: auto uses sdpa on CUDA (fast) and eager on CPU; flash_attention_2 requires flash-attn.',
    )
    parser.add_argument(
        '--compile_model',
        action='store_true',
        help='Wrap the model with torch.compile (PyTorch 2+; first steps are slow, then often faster).',
    )
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load validation dataset
    print(f"Loading validation dataset from: {args.validation_data}")
    with open(args.validation_data, 'r') as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} validation examples")
    
    # Load model and tokenizer
    attn_impl = None if args.attn_implementation == 'auto' else args.attn_implementation
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        adapter_path=args.adapter_path,
        attn_implementation=attn_impl,
        compile_model=args.compile_model,
    )
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process dataset
    print(f"\nGenerating predictions...")
    process_dataset(
        model,
        tokenizer,
        dataset,
        args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_instruction_format=not args.use_chatgpt_format,
        extract_skeleton=args.extract_skeleton,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
    )
    
    print(f"\n✅ Done! Predictions saved to: {args.output}")
    print(f"\nTo evaluate predictions, run:")
    print(f"  cd eval")
    print(f"  python evaluation.py --input {args.output} --gold data/gold.txt --db data/database --etype exec")

if __name__ == "__main__":
    main()

