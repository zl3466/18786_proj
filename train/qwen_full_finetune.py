"""
Qwen Full Parameter Finetuning Script

This script implements full parameter finetuning for Qwen models on text-to-SQL tasks.
All model parameters are updated during training (no LoRA, no quantization).
Requires significant VRAM (recommended: 40GB+ for 7B model).
"""

import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import os
from typing import Optional

class QwenFullConfig:
    """Configuration class for Qwen full parameter training"""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    TRUST_REMOTE_CODE = True
    
    # Training settings
    OUTPUT_DIR = "./qwen-full-finetuned"
    PER_DEVICE_BATCH_SIZE = 1  # Very small due to full parameter training
    GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 16
    LEARNING_RATE = 1e-5  # Lower LR for full fine-tuning
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1
    MAX_STEPS = 3000
    MAX_GRAD_NORM = 1.0
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    EVAL_STEPS = 500
    SAVE_TOTAL_LIMIT = 3
    MAX_SEQ_LENGTH = 2048
    
    # Advanced settings
    USE_GRADIENT_CHECKPOINTING = True  # Critical for memory savings
    GROUP_BY_LENGTH = True
    FP16 = True  # Can use bf16 if supported
    BF16 = False  # Set to True if your GPU supports bfloat16
    USE_DEEPSPEED = False  # Enable for multi-GPU or ZeRO optimization

def load_qwen_model(config: QwenFullConfig):
    """Load Qwen model for full parameter training"""
    print(f"Loading model: {config.MODEL_NAME} (full parameter training)")
    print("WARNING: This requires significant VRAM (40GB+ recommended for 7B model)")
    
    # Load model in full precision (or mixed precision)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        device_map="auto",
        torch_dtype=torch.float16 if config.FP16 else torch.float32,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Enable gradient checkpointing (CRITICAL for memory)
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (saves ~50% memory)")
    
    return model

def setup_tokenizer(config: QwenFullConfig, model):
    """Setup tokenizer for Qwen"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return tokenizer

def load_training_data(data_path: str, use_curriculum: bool = False):
    """Load and optionally prepare training data with curriculum learning"""
    print(f"Loading training data from: {data_path}")
    
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Convert to dataset
    dataset = Dataset.from_list(data)
    
    # Apply curriculum learning if requested
    if use_curriculum:
        print("Applying curriculum learning (sorting by complexity)...")
        dataset = sort_by_complexity(dataset)
    
    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=42)
    
    return dataset

def sort_by_complexity(dataset):
    """Sort dataset by SQL query complexity for curriculum learning"""
    def get_complexity_score(example):
        sql = example.get('ground_truth', example.get('text', ''))
        if isinstance(sql, dict):
            sql = sql.get('sql', '')
        
        score = 0
        sql_upper = str(sql).upper()
        if 'JOIN' in sql_upper:
            score += sql_upper.count('JOIN') * 2
        if 'GROUP BY' in sql_upper:
            score += 1
        if 'HAVING' in sql_upper:
            score += 1
        if 'UNION' in sql_upper or 'INTERSECT' in sql_upper:
            score += 2
        if sql_upper.count('SELECT') > 1:
            score += 2
        if 'ORDER BY' in sql_upper:
            score += 1
        return score
    
    # Add complexity score
    dataset = dataset.map(lambda x: {'complexity': get_complexity_score(x)})
    
    # Sort by complexity
    dataset = dataset.sort('complexity')
    
    return dataset

def create_training_arguments(config: QwenFullConfig, eval_dataset: Optional[Dataset] = None):
    """Create training arguments"""
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_32bit",  # Can use "adamw_torch" for full precision
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        max_steps=config.MAX_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        fp16=config.FP16 and not config.BF16,
        bf16=config.BF16,
        report_to='tensorboard',
        group_by_length=config.GROUP_BY_LENGTH,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=config.EVAL_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        # DeepSpeed config if enabled
        deepspeed="deepspeed_config.json" if config.USE_DEEPSPEED else None,
    )
    
    return training_args

def main(
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    config: Optional[QwenFullConfig] = None,
    use_curriculum: bool = False,
):
    """Main training function"""
    
    if config is None:
        config = QwenFullConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model = load_qwen_model(config)
    tokenizer = setup_tokenizer(config, model)
    
    # Load training data
    train_dataset = load_training_data(train_data_path, use_curriculum=use_curriculum)
    
    # Load eval data if provided
    eval_dataset = None
    if eval_data_path:
        eval_dataset = load_training_data(eval_data_path)
    
    # Create training arguments
    training_args = create_training_arguments(config, eval_dataset)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Train
    print("Starting full parameter training...")
    print("NOTE: This will update all model parameters. Ensure you have sufficient VRAM.")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    
    # Create model card
    trainer.create_model_card()
    
    print(f"Training completed! Model saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full parameter finetune Qwen model")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON or JSONL)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (optional)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-full-finetuned",
        help="Output directory for model"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (lower for full fine-tuning)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per device batch size (keep small for full training)"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 instead of float16 (if GPU supports it)"
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use DeepSpeed for multi-GPU or ZeRO optimization"
    )
    
    args = parser.parse_args()
    
    # Create config with command-line arguments
    config = QwenFullConfig()
    config.MODEL_NAME = args.model_name
    config.OUTPUT_DIR = args.output_dir
    config.MAX_STEPS = args.max_steps
    config.LEARNING_RATE = args.learning_rate
    config.PER_DEVICE_BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    config.BF16 = args.bf16
    config.FP16 = not args.bf16
    config.USE_DEEPSPEED = args.deepspeed
    
    main(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        config=config,
        use_curriculum=args.curriculum,
    )










