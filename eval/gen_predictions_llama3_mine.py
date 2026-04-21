"""
Generate SQL predictions using Llama3-7B model on Spider validation dataset

This script loads Llama3-7B from Hugging Face, generates SQL queries
for the validation dataset, and saves predictions for evaluation.
"""

import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

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
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    generated_text = generated_text[len(prompt):].strip()
    
    # Stop at certain sequences
    if stop_sequences:
        for stop_seq in stop_sequences:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0].strip()
    
    return generated_text

def load_model_and_tokenizer(model_name: str, device: str = "auto", load_in_8bit: bool = False):
    """
    Load Llama3 model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load on ("auto", "cuda", "cpu")
        load_in_8bit: Whether to use 8-bit quantization
        
    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"   Model dtype: {model.dtype}")
    
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
    """
    # Check if we're resuming
    existing_predictions = []
    if resume_from > 0 and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_predictions = [line.strip() for line in f]
        print(f"Resuming from index {resume_from}, found {len(existing_predictions)} existing predictions")
    
    # Open output file in append mode if resuming
    mode = 'a' if resume_from > 0 else 'w'
    
    with open(output_file, mode) as f:
        # Process each entry
        for i, entry in enumerate(tqdm(dataset, desc="Generating predictions")):
            # Skip if already processed
            if i < resume_from:
                continue
            
            try:
                # Build prompt
                prompt = build_prompt(
                    entry['question'],
                    entry['db_info'],
                    use_instruction_format=use_instruction_format
                )
                
                # Generate SQL
                raw_response = generate_sql(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    stop_sequences=["###", "\n\n\n"],
                )
                
                # Format response
                sql = format_sql_response(raw_response, extract_skeleton=extract_skeleton)
                
                # Write to file
                f.write(sql + "\n")
                f.flush()  # Ensure it's written immediately
                
            except Exception as e:
                print(f"\n❌ Error processing index {i}: {e}")
                # Write empty line as placeholder
                f.write("\n")
                f.flush()
                continue
    
    print(f"\n✅ Predictions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL predictions using Llama3-7B on Spider validation dataset"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='HuggingFace model name or local path. Options: meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-8B, or local path'
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
    
    args = parser.parse_args()
    
    # Load validation dataset
    print(f"Loading validation dataset from: {args.validation_data}")
    with open(args.validation_data, 'r') as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} validation examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        load_in_8bit=args.load_in_8bit
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
    )
    
    print(f"\n✅ Done! Predictions saved to: {args.output}")
    print(f"\nTo evaluate predictions, run:")
    print(f"  cd eval")
    print(f"  python evaluation.py --input {args.output} --gold data/gold.txt --db data/database --etype exec")

if __name__ == "__main__":
    main()
