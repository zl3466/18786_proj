# Finetuning Methods Comparison: LoRA vs Full Parameter

This guide compares LoRA (no quantization) and Full Parameter finetuning for Qwen 2.5 7B.

## Quick Comparison

| Aspect | LoRA (No Quantization) | Full Parameter |
|--------|----------------------|----------------|
| **VRAM Required** | ~16-24GB | ~40GB+ |
| **Training Speed** | Fast | Slower |
| **Model Size** | Small adapter (~100MB) | Full model (~14GB) |
| **Flexibility** | Can switch adapters | Single model |
| **Performance** | Very good (95%+ of full) | Best possible |
| **Best For** | Most use cases | Maximum accuracy |

---

## Method 1: LoRA (No Quantization)

### Script: `train/qwen_lora_finetune.py`

**Characteristics:**
- ✅ Full precision model (no quantization)
- ✅ Only trains adapter weights (~0.1% of parameters)
- ✅ Much lower VRAM requirement
- ✅ Fast training
- ✅ Can have multiple adapters for different tasks

### Usage:

```bash
python train/qwen_lora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen-lora-adapter \
    --max_steps 3000 \
    --batch_size 2 \
    --gradient_accumulation 8
```

### Configuration:

- **LoRA Rank**: 64 (can reduce to 32 for less VRAM)
- **LoRA Alpha**: 32
- **Batch Size**: 2 per device (effective: 16 with gradient accumulation)
- **Learning Rate**: 2e-4
- **VRAM**: ~16-24GB

### Advantages:

1. **Memory Efficient**: Only trains adapter weights
2. **Fast**: Quick training iterations
3. **Flexible**: Can switch between different adapters
4. **Good Performance**: Typically 95%+ of full fine-tuning
5. **No Quantization**: Full precision model

### When to Use:

- ✅ Limited VRAM (< 40GB)
- ✅ Want to experiment with different adapters
- ✅ Need fast iteration
- ✅ Most production use cases

---

## Method 2: Full Parameter Finetuning

### Script: `train/qwen_full_finetune.py`

**Characteristics:**
- ✅ Updates all model parameters
- ✅ Maximum model capacity
- ✅ Best possible performance
- ⚠️ Requires significant VRAM
- ⚠️ Slower training

### Usage:

#### Single GPU (if you have 40GB+ VRAM):

```bash
python train/qwen_full_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen-full-finetuned \
    --max_steps 3000 \
    --batch_size 1 \
    --gradient_accumulation 16
```

#### Multi-GPU with DeepSpeed:

```bash
deepspeed --num_gpus=2 train/qwen_full_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen-full-finetuned \
    --deepspeed \
    --batch_size 1 \
    --gradient_accumulation 8
```

### Configuration:

- **Batch Size**: 1 per device (very small due to memory)
- **Gradient Accumulation**: 16 (to maintain effective batch size)
- **Learning Rate**: 1e-5 (lower than LoRA)
- **Gradient Checkpointing**: Enabled (critical for memory)
- **VRAM**: ~40GB+ for 7B model

### Advantages:

1. **Maximum Performance**: Best possible accuracy
2. **Full Capacity**: All parameters updated
3. **No Adapter Overhead**: Direct model updates
4. **Research**: Best for understanding full model behavior

### When to Use:

- ✅ Have 40GB+ VRAM (or multiple GPUs)
- ✅ Need maximum accuracy
- ✅ Research/experimentation
- ✅ Production where accuracy is critical

### Memory Optimization Tips:

1. **Use DeepSpeed ZeRO**: Offloads optimizer states to CPU
2. **Gradient Checkpointing**: Enabled by default (saves ~50% memory)
3. **Small Batch Size**: Use batch_size=1 with high gradient accumulation
4. **Mixed Precision**: Use fp16 or bf16
5. **Multi-GPU**: Distribute across multiple GPUs

---

## Performance Comparison

### Expected Results (Text-to-SQL):

| Method | Execution Accuracy | Training Time | VRAM |
|--------|-------------------|--------------|------|
| **LoRA** | ~60-65% | ~2-4 hours | 16-24GB |
| **Full Parameter** | ~62-67% | ~4-8 hours | 40GB+ |
| **QLoRA** | ~59-64% | ~2-4 hours | 8-12GB |

*Note: Results vary based on dataset, hyperparameters, and training duration*

### Key Insights:

1. **LoRA is usually sufficient**: 95%+ of full fine-tuning performance
2. **Full fine-tuning**: 2-5% improvement, but requires 2-3x more resources
3. **Diminishing returns**: Full fine-tuning may not be worth the cost for most use cases

---

## Memory Requirements

### LoRA Training:

```
Base Model (FP16):        ~14GB
LoRA Adapter:             ~0.1GB
Optimizer States:         ~0.5GB
Activations:              ~2-4GB
Gradient Checkpointing:   Saves ~50%
─────────────────────────────────────
Total VRAM:               ~16-24GB
```

### Full Parameter Training:

```
Base Model (FP16):        ~14GB
Optimizer States:         ~28GB (AdamW)
Activations:              ~4-8GB
Gradient Checkpointing:   Saves ~50%
─────────────────────────────────────
Total VRAM:               ~40-50GB
```

**With DeepSpeed ZeRO Stage 2:**
- Offloads optimizer states to CPU
- Reduces VRAM to ~20-25GB per GPU
- Requires multiple GPUs

---

## Recommendation Matrix

### Choose **LoRA** if:

- ✅ You have 16-24GB VRAM
- ✅ You want fast iteration
- ✅ You need good performance (95%+ of full)
- ✅ You want to experiment with multiple adapters
- ✅ **Most use cases** ← **Recommended**

### Choose **Full Parameter** if:

- ✅ You have 40GB+ VRAM (or multiple GPUs)
- ✅ You need maximum accuracy
- ✅ You're doing research
- ✅ Performance is critical and resources are available

---

## Quick Start Commands

### LoRA Training (Recommended):

```bash
# Basic training
python train/qwen_lora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --output_dir ./qwen-lora-adapter

# With custom settings
python train/qwen_lora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --output_dir ./qwen-lora-adapter \
    --batch_size 2 \
    --gradient_accumulation 8 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --max_steps 3000
```

### Full Parameter Training:

```bash
# Single GPU (requires 40GB+ VRAM)
python train/qwen_full_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --output_dir ./qwen-full-finetuned \
    --batch_size 1 \
    --gradient_accumulation 16

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=2 train/qwen_full_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --output_dir ./qwen-full-finetuned \
    --deepspeed \
    --batch_size 1 \
    --gradient_accumulation 8
```

---

## Loading Trained Models

### LoRA Adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "./qwen-lora-adapter/checkpoint-3000"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)
```

### Full Parameter Model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load directly (no adapter needed)
model = AutoModelForCausalLM.from_pretrained(
    "./qwen-full-finetuned/checkpoint-3000",
    trust_remote_code=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen-full-finetuned/checkpoint-3000",
    trust_remote_code=True
)
```

---

## Summary

**For most users: Use LoRA (no quantization)**
- ✅ Best balance of performance and resources
- ✅ Fast training
- ✅ Good accuracy
- ✅ Flexible

**Only use Full Parameter if:**
- You have 40GB+ VRAM
- You need maximum accuracy
- You're doing research

Both scripts are ready to use with your `train_sql_skeleton.jsonl` dataset!










