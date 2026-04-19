# Alternative Finetuning Methods for Qwen 2.5 7B

This guide covers different methods to finetune Qwen 2.5 7B on your SQL skeleton dataset, beyond LlamaFactory.

## Method Comparison

| Method                        | Pros                                    | Cons                 | Best For              |
| ----------------------------- | --------------------------------------- | -------------------- | --------------------- |
| **Transformers + PEFT + TRL** | Full control, flexible, well-documented | More code to write   | Custom training logic |
| **Axolotl**                   | Simple config, many features            | Less popular         | Quick setup           |
| **Unsloth**                   | Very fast, memory efficient             | Newer, less tested   | Speed optimization    |
| **DeepSpeed**                 | Multi-GPU, very efficient               | Complex setup        | Large-scale training  |
| **Jupyter Notebook**          | Interactive, easy debugging             | Not production-ready | Experimentation       |

---

## Method 1: Transformers + PEFT + TRL (Recommended)

**Status**: ✅ Already implemented in `train/qwen_qlora_finetune.py`

### Advantages:

- Full control over training process
- Industry standard (HuggingFace ecosystem)
- Well-documented and widely used
- Easy to customize and debug
- Works with your existing data format

### Usage:

```bash
python train/qwen_qlora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --eval_data data/validation_sql_skeleton.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen-skeleton-qlora-adapter \
    --max_steps 3000 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --gradient_accumulation 4
```

### Key Features:

- QLoRA with 4-bit quantization
- LoRA rank=64, alpha=32
- SFTTrainer for supervised fine-tuning
- Gradient checkpointing
- Curriculum learning option

**This is the most flexible and recommended approach.**

---

## Method 2: Axolotl

**Axolotl** is a unified training framework similar to LlamaFactory but with different features.

### Installation:

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

### Configuration (`train/axolotl_config.yaml`):

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
model_type: QwenForCausalLM
load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: json
    data_files: data/train_sql_skeleton.jsonl
    type: sharegpt

dataset_prepared_path: ./axolotl_cache
val_set_size: 0.1
output_dir: ./qwen-skeleton-axolotl

adapter: lora
lora_model_dir:
lora_r: 64
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 2.0e-4
warmup_steps: 100
train_on_inputs: false
group_by_length: true
bf16: auto
fp16: false
tf32: false
gradient_checkpointing: true
logging_steps: 10
save_steps: 500
```

### Usage:

```bash
accelerate launch -m axolotl.cli.train train/axolotl_config.yaml
```

### Advantages:

- Simple YAML configuration
- Good for multi-GPU setups
- Active development

---

## Method 3: Unsloth (Fastest)

**Unsloth** is optimized for speed and memory efficiency.

### Installation:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

### Usage Script (`train/unsloth_finetune.py`):

```python
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto detection
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# Load dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="data/train_sql_skeleton.jsonl", split="train")

# Tokenize
def formatting_prompts_func(examples):
    texts = examples["text"]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=3000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
    ),
)

# Train
trainer.train()

# Save
model.save_pretrained("qwen-skeleton-unsloth")
```

### Advantages:

- **2-5x faster** than standard training
- More memory efficient
- Easy to use

### Usage:

```bash
python train/unsloth_finetune.py
```

---

## Method 4: DeepSpeed (Multi-GPU)

**DeepSpeed** is best for multi-GPU training with ZeRO optimization.

### Installation:

```bash
pip install deepspeed
```

### Configuration (`train/deepspeed_config.json`):

```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 0.3,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 100
    }
  },
  "fp16": {
    "enabled": true
  },
  "wall_clock_breakdown": false
}
```

### Usage:

```bash
deepspeed --num_gpus=2 train/qwen_qlora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --deepspeed train/deepspeed_config.json
```

### Advantages:

- Best for multi-GPU setups
- ZeRO optimization saves memory
- Can train larger models

---

## Method 5: Jupyter Notebook (Interactive)

**Status**: ✅ Already exists as `train/hf_qlora_wizard_coder.ipynb`

### Advantages:

- Interactive development
- Easy to experiment
- Visual debugging
- Step-by-step execution

### Usage:

1. Open `train/hf_qlora_wizard_coder.ipynb`
2. Modify for Qwen model
3. Run cells interactively

### Best for:

- Experimentation
- Learning
- Debugging

---

## Method 6: Custom Training Loop (Maximum Control)

For ultimate control, write your own training loop:

```python
# train/custom_training.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load model
model = AutoModelForCausalLM.from_pretrained(...)
model = get_peft_model(model, LoraConfig(...))

# Custom training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
for epoch in range(3):
    for batch in tqdm(dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Advantages:

- Complete control
- Custom loss functions
- Custom optimizers
- Research flexibility

---

## Recommendation Matrix

### Choose **Transformers + PEFT + TRL** if:

- ✅ You want full control
- ✅ You need custom training logic
- ✅ You want industry-standard approach
- ✅ You're already familiar with HuggingFace

### Choose **Unsloth** if:

- ✅ Speed is critical (2-5x faster)
- ✅ You have limited time
- ✅ You want simple setup

### Choose **Axolotl** if:

- ✅ You prefer YAML configs
- ✅ You need multi-GPU support
- ✅ You want active community

### Choose **DeepSpeed** if:

- ✅ You have multiple GPUs
- ✅ You're training very large models
- ✅ Memory is a constraint

### Choose **Jupyter Notebook** if:

- ✅ You're experimenting
- ✅ You want interactive debugging
- ✅ You're learning

---

## Quick Start: Transformers + PEFT + TRL

Since you already have `train/qwen_qlora_finetune.py`, this is the easiest:

```bash
# Install dependencies (already in requirements.txt)
pip install transformers peft trl datasets bitsandbytes accelerate

# Run training
python train/qwen_qlora_finetune.py \
    --train_data data/train_sql_skeleton.jsonl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen-skeleton-qlora-adapter \
    --max_steps 3000
```

This gives you:

- ✅ Full control
- ✅ Well-tested code
- ✅ Easy to customize
- ✅ Production-ready

---

## Performance Comparison

| Method                | Training Speed   | Memory Usage | Setup Complexity | Flexibility |
| --------------------- | ---------------- | ------------ | ---------------- | ----------- |
| Transformers+PEFT+TRL | Medium           | Medium       | Medium           | ⭐⭐⭐⭐⭐  |
| Unsloth               | ⭐⭐⭐⭐⭐       | Low          | Low              | ⭐⭐⭐      |
| Axolotl               | Medium           | Medium       | Low              | ⭐⭐⭐⭐    |
| DeepSpeed             | High (multi-GPU) | Very Low     | High             | ⭐⭐⭐      |
| LlamaFactory          | Medium           | Medium       | Very Low         | ⭐⭐⭐      |

---

## Next Steps

1. **Start with existing code**: Use `train/qwen_qlora_finetune.py` (already set up)
2. **Try Unsloth**: If you need speed, try the Unsloth approach
3. **Scale up**: Use DeepSpeed if you have multiple GPUs

All methods will produce similar results - choose based on your needs for speed, control, and setup complexity.









