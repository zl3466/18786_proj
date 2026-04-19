"""
python train/qwen_lora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --output_dir ./qwen-lora-adapter \
    --max_steps 3000
"""

import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import os
from typing import Optional

class QwenLoraConfig:
    """Configuration class for Qwen LoRA training (no quantization)"""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    TRUST_REMOTE_CODE = True
    
    # LoRA settings
    LORA_R = 64  # Rank - higher for better capacity
    LORA_ALPHA = 32  # LoRA alpha (typically 2x r)
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ]
    
    # Training settings
    OUTPUT_DIR = "./qwen-lora-adapter"
    PER_DEVICE_BATCH_SIZE = 2  # Lower than QLoRA due to no quantization
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 16
    LEARNING_RATE = 2e-4
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1
    MAX_STEPS = 3000
    MAX_GRAD_NORM = 0.3
    SAVE_STEPS = 500
    LOGGING_STEPS = 10
    EVAL_STEPS = 500
    SAVE_TOTAL_LIMIT = 3
    MAX_SEQ_LENGTH = 2048
    
    # Advanced settings
    USE_GRADIENT_CHECKPOINTING = True
    GROUP_BY_LENGTH = True
    FP16 = True  # Can use bf16 if supported
    BF16 = False  # Set to True if your GPU supports bfloat16

    # Weights & Biases (https://wandb.ai/<entity>/<project>)
    USE_WANDB = True
    WANDB_ENTITY: Optional[str] = None
    WANDB_PROJECT: Optional[str] = None
    WANDB_RUN_NAME: Optional[str] = None

class PeftSavingCallback(TrainerCallback):
    """Callback to save PEFT adapters correctly"""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)
        
        # Remove pytorch_model.bin if it exists (not needed for PEFT)
        pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

def load_qwen_model(config: QwenLoraConfig):
    # BF16 weights + bf16 training; FP16 mixed precision needs FP32 weights (fp16 weights + GradScaler
    # breaks at clip_grad_norm_: "Attempting to unscale FP16 gradients").
    if config.BF16:
        torch_dtype = torch.bfloat16
    elif config.FP16:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Enable gradient checkpointing if requested
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    return model

def setup_tokenizer(config: QwenLoraConfig, model):
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

def setup_lora(config: QwenLoraConfig, model):
    """Configure and apply LoRA"""
    print("Setting up LoRA configuration...")
    
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

def load_training_data(data_path: str, is_eval: bool = False):
    """Load and prepare training/eval data
    
    Args:
        data_path: Path to JSON/JSONL file
        is_eval: If True, convert validation format to training format
    """
    print(f"Loading data from: {data_path}")
    
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
    
    # Convert validation format to training format if needed
    if is_eval:
        converted_data = []
        for item in data:
            # Check if it's already in training format (has 'text' field)
            if 'text' in item:
                converted_data.append(item)
            else:
                # Convert validation format to training format
                # Validation format: {"db_id": "...", "question": "...", "db_info": "...", "ground_truth": "..."}
                # Training format: {"db_id": "...", "text": "### Instruction:\n...\n### Response:\n..."}
                question = item.get('question', '')
                db_info = item.get('db_info', '')
                ground_truth = item.get('ground_truth', '')
                
                # Reconstruct the instruction-response format
                text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\nConvert text to sql: {question}{db_info}\n\n### Response:\n\n{ground_truth}"
                
                converted_data.append({
                    "db_id": item.get('db_id', ''),
                    "text": text
                })
        data = converted_data
    
    # Convert to dataset
    dataset = Dataset.from_list(data)
    
    return dataset

def create_training_arguments(config: QwenLoraConfig, eval_dataset: Optional[Dataset] = None):
    """Create training arguments"""
    if config.USE_WANDB:
        report_to = ["tensorboard", "wandb"]
    else:
        report_to = "tensorboard"
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
        fp16=config.FP16 and not config.BF16,
        bf16=config.BF16,
        report_to=report_to,
        run_name=config.WANDB_RUN_NAME if config.USE_WANDB else None,
        group_by_length=config.GROUP_BY_LENGTH,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.EVAL_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
    )
    
    return training_args

def main(
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    config: Optional[QwenLoraConfig] = None,
):
    """Main training function"""
    
    if config is None:
        config = QwenLoraConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if config.USE_WANDB:
        if config.WANDB_PROJECT:
            os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
        if config.WANDB_ENTITY:
            os.environ.setdefault("WANDB_ENTITY", config.WANDB_ENTITY)
    
    # Load model and tokenizer
    model = load_qwen_model(config)
    tokenizer = setup_tokenizer(config, model)
    
    # Setup LoRA
    model = setup_lora(config, model)
    
    # Upcast layer norms for stability
    upcast_layer_norms(model)
    
    # Load training data
    train_dataset = load_training_data(train_data_path, is_eval=False)
    
    # Create training arguments (no evaluation)
    training_args = create_training_arguments(config, eval_dataset=None)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,  # No evaluation during training
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
    
    parser = argparse.ArgumentParser(description="Finetune Qwen model with LoRA (no quantization)")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON or JSONL)"
    )
    # Removed eval_data argument - no evaluation during training
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-lora-adapter",
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
        "--batch_size",
        type=int,
        default=2,
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 instead of float16 (if GPU supports it)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user), see https://wandb.ai/<entity>/<project>",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging (tensorboard only)",
    )
    
    args = parser.parse_args()
    
    # Create config with command-line arguments
    config = QwenLoraConfig()
    config.MODEL_NAME = args.model_name
    config.OUTPUT_DIR = args.output_dir
    config.MAX_STEPS = args.max_steps
    config.LEARNING_RATE = args.learning_rate
    config.PER_DEVICE_BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    config.LORA_R = args.lora_rank
    config.LORA_ALPHA = args.lora_alpha
    config.BF16 = args.bf16
    config.FP16 = not args.bf16
    config.USE_WANDB = not args.no_wandb
    config.WANDB_ENTITY = args.wandb_entity
    config.WANDB_PROJECT = args.wandb_project
    config.WANDB_RUN_NAME = args.run_name
    
    main(
        train_data_path=args.train_data,
        eval_data_path=None,  # No evaluation during training
        config=config
    )
