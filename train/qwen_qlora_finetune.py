"""
Qwen QLoRA Finetuning Script with Advanced Techniques

This script implements QLoRA finetuning for Qwen models on text-to-SQL tasks
with advanced techniques for improved accuracy.
"""

import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import os
from typing import Optional

class QwenQloraConfig:
    """Configuration class for Qwen QLoRA training"""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Can be changed to other Qwen models
    TRUST_REMOTE_CODE = True
    
    # Quantization settings
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = torch.float16
    USE_DOUBLE_QUANT = True  # Double quantization for better compression
    
    # LoRA settings
    LORA_R = 64  # Rank - higher for better capacity
    LORA_ALPHA = 32  # LoRA alpha (typically 2x r)
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    
    # Training settings
    OUTPUT_DIR = "./qwen-qlora-adapter"
    PER_DEVICE_BATCH_SIZE = 4  # Adjust based on VRAM
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
    LEARNING_RATE = 2e-4
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1
    MAX_STEPS = 3000
    MAX_GRAD_NORM = 0.3
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    EVAL_STEPS = 500
    SAVE_TOTAL_LIMIT = 3
    MAX_SEQ_LENGTH = 2048  # Increase if needed
    
    # Advanced settings
    USE_GRADIENT_CHECKPOINTING = True
    GROUP_BY_LENGTH = True
    FP16 = True

class PeftSavingCallback(TrainerCallback):
    """Callback to save PEFT adapters correctly"""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)
        
        # Remove pytorch_model.bin if it exists (not needed for PEFT)
        pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

def load_qwen_model(config: QwenQloraConfig):
    """Load Qwen model with 4-bit quantization"""
    print(f"Loading model: {config.MODEL_NAME}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=config.USE_DOUBLE_QUANT,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        device_map="auto",
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Enable gradient checkpointing if requested
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    return model

def setup_tokenizer(config: QwenQloraConfig, model):
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

def setup_lora(config: QwenQloraConfig, model):
    """Configure and apply LoRA"""
    print("Setting up LoRA configuration...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def upcast_layer_norms(model):
    """Upcast layer norms to float32 for stable training"""
    print("Upcasting layer norms to float32...")
    for name, module in model.named_modules():
        if "norm" in name.lower():
            module = module.to(torch.float32)

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

def create_training_arguments(config: QwenQloraConfig, eval_dataset: Optional[Dataset] = None):
    """Create training arguments"""
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_32bit",
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        max_steps=config.MAX_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        fp16=config.FP16,
        report_to='tensorboard',
        group_by_length=config.GROUP_BY_LENGTH,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=config.EVAL_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
    )
    
    return training_args

def main(
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    config: Optional[QwenQloraConfig] = None,
    use_curriculum: bool = False,
):
    """Main training function"""
    
    if config is None:
        config = QwenQloraConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model = load_qwen_model(config)
    tokenizer = setup_tokenizer(config, model)
    
    # Setup LoRA
    model = setup_lora(config, model)
    
    # Upcast layer norms for stability
    upcast_layer_norms(model)
    
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
        peft_config=None,  # Already applied
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[PeftSavingCallback],
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    
    # Create model card
    trainer.create_model_card()
    
    print(f"Training completed! Model saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune Qwen model with QLoRA")
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
        default="./qwen-qlora-adapter",
        help="Output directory for adapter"
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
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    args = parser.parse_args()
    
    # Create config with command-line arguments
    config = QwenQloraConfig()
    config.MODEL_NAME = args.model_name
    config.OUTPUT_DIR = args.output_dir
    config.MAX_STEPS = args.max_steps
    config.LEARNING_RATE = args.learning_rate
    config.PER_DEVICE_BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    
    main(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        config=config,
        use_curriculum=args.curriculum,
    )










