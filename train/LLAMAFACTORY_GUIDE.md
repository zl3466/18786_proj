# LlamaFactory Finetuning Guide for Qwen 2.5 7B

This guide explains how to use LlamaFactory to finetune Qwen 2.5 7B on the SQL skeleton dataset.

## Installation

1. Install LlamaFactory:
```bash
pip install llamafactory
```

Or add to requirements.txt:
```bash
pip install -r requirements.txt
```

## Dataset Format

The training data (`data/train_sql_skeleton.jsonl`) is already in the correct format:
- Each line is a JSON object with a `text` field
- The `text` field contains the full instruction-response format:
  ```
  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

  ### Instruction:

  Convert text to sql: [question] | [schema]

  ### Response:

  [skeleton | actual SQL]
  ```

## Configuration Files

### 1. Training Config: `train/qwen_skeleton_llamafactory.yaml`

This file contains all training parameters:
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Method**: LoRA (QLoRA with 4-bit quantization)
- **LoRA Settings**: rank=64, alpha=32, dropout=0.1
- **Training**: batch_size=4, gradient_accumulation=4, learning_rate=2e-4
- **Output**: `./qwen-skeleton-qlora-adapter`

### 2. Dataset Info: `train/dataset_info.json`

This file tells LlamaFactory how to parse your dataset:
- **Format**: alpaca (instruction-response format)
- **File**: `../data/train_sql_skeleton.jsonl`
- **Columns**: Uses `text` field directly

## Running Training

### Option 1: Command Line

```bash
cd train
llamafactory-cli train qwen_skeleton_llamafactory.yaml
```

### Option 2: Using Script

```bash
bash train/run_llamafactory.sh
```

### Option 3: Python API

```python
from llamafactory import TrainArgument, run_exp

args = TrainArgument(
    stage="sft",
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    dataset="sql_skeleton",
    dataset_dir="train",
    template="default",
    finetuning_type="lora",
    output_dir="./qwen-skeleton-qlora-adapter",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2.0e-4,
    num_train_epochs=3.0,
    max_steps=3000,
    quantization_bit=4,
    lora_rank=64,
    lora_alpha=32,
)

run_exp(args)
```

## Training Parameters Explained

### LoRA Configuration
- `lora_rank: 64` - Higher rank for better capacity (can reduce to 32 for less VRAM)
- `lora_alpha: 32` - Typically 2x the rank value
- `lora_dropout: 0.1` - Regularization to prevent overfitting

### Quantization
- `quantization_bit: 4` - 4-bit quantization (QLoRA)
- `quantization_type: nf4` - NormalFloat4 quantization
- `compute_dtype: float16` - Computation in float16

### Training
- `per_device_train_batch_size: 4` - Adjust based on VRAM
- `gradient_accumulation_steps: 4` - Effective batch size = 16
- `learning_rate: 2.0e-4` - Standard for QLoRA
- `max_steps: 3000` - Can adjust based on dataset size

## Monitoring Training

Training logs will show:
- Loss at each logging step
- Training progress
- Model checkpoints saved every 500 steps

Checkpoints are saved in: `./qwen-skeleton-qlora-adapter/checkpoint-{step}`

## Using the Fine-tuned Model

After training, load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./qwen-skeleton-qlora-adapter/checkpoint-3000"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)
```

## Adjusting for Your Hardware

### Low VRAM (< 16GB):
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
lora_rank: 32
```

### Medium VRAM (16-24GB):
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
lora_rank: 64
```

### High VRAM (> 24GB):
```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
lora_rank: 64
```

## Troubleshooting

### Issue: "Dataset not found"
- Make sure `dataset_info.json` has the correct path to your JSONL file
- Path is relative to where you run the command

### Issue: "Out of memory"
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `lora_rank` to 32

### Issue: "Template not found"
- Make sure you're using `template: default` for Qwen models
- Or use `template: qwen` if available

## Comparison with Previous Setup

| Feature | Transformers+PEFT+TRL | LlamaFactory |
|---------|----------------------|--------------|
| **Code Lines** | ~375 lines | ~50 lines config |
| **Setup Time** | More setup | Quick config |
| **Flexibility** | Full control | Pre-configured |
| **Web UI** | No | Yes (optional) |

## Next Steps

1. Run training: `llamafactory-cli train train/qwen_skeleton_llamafactory.yaml`
2. Monitor training progress
3. Evaluate on validation set
4. Use the fine-tuned model for inference

For evaluation, see `eval/gen_predictions_llama3.py` as a reference for creating prediction scripts.










