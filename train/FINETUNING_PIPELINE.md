# Text-to-SQL Finetuning Pipeline with Skeleton Data and Schema Linking

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINETUNING PIPELINE                                 │
│                    (Skeleton Data + Schema Linking)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA PREPARATION                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw Spider Dataset
           │
           ▼
    ┌─────────────────┐
    │ Preprocessing   │
    │ - Extract SQL   │
    │ - Generate      │
    │   skeletons     │
    └─────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ train_sql_skeleton.jsonl                                │
    │ Format:                                                  │
    │ {                                                        │
    │   "text": "### Instruction:\n...\n### Response:\n      │
    │            skeleton | actual_sql"                        │
    │ }                                                        │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Convert to OpenAI Format (Optional)                     │
    │ train_sql_skeleton_openai.jsonl                         │
    │ Format:                                                  │
    │ {                                                        │
    │   "messages": [                                          │
    │     {"role": "system", ...},                            │
    │     {"role": "user", ...},                               │
    │     {"role": "assistant", "content": "skeleton | sql"}  │
    │   ]                                                      │
    │ }                                                        │
    └─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: SCHEMA RANKING (For Inference)                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Validation Dataset
           │
           ▼
    ┌─────────────────┐
    │ Schema Linking  │
    │ - Rank tables   │
    │ - Rank columns  │
    │ - Remove        │
    │   irrelevant    │
    └─────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ validation_sql_ranked.json                              │
    │ - Tables ordered by relevance                           │
    │ - Columns ordered by relevance                          │
    │ - Irrelevant tables removed                             │
    └─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MODEL FINETUNING                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    Base Model: Qwen/Qwen2.5-7B-Instruct
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Training Configuration                                 │
    │ - Method: LoRA (or QLoRA/Full)                         │
    │ - Data: train_sql_skeleton_openai.jsonl                │
    │ - Format: OpenAI messages                               │
    │ - Loss: Only on assistant messages                     │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Training Process                                        │
    │                                                         │
    │  Input: "### Instruction:\n...\n### Response:\n..."   │
    │         ↓                                               │
    │  Tokenize & Mask                                        │
    │         ↓                                               │
    │  Instruction tokens → masked (-100)                     │
    │  Response tokens → loss computed                        │
    │         ↓                                               │
    │  Update LoRA weights                                    │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Fine-tuned Model                                        │
    │ - LoRA Adapter: ./qwen-skeleton-qlora-adapter         │
    │ - Learned: Generate skeleton | actual_sql format       │
    └─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: INFERENCE WITH SCHEMA LINKING                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    User Question
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Schema Linking (Before Inference)                       │
    │                                                         │
    │  1. Load database schema                                │
    │  2. Rank tables by relevance to question               │
    │  3. Rank columns by relevance                           │
    │  4. Remove irrelevant tables/columns                    │
    │  5. Format schema (clear context format)                 │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Formatted Input                                         │
    │                                                         │
    │  "### Complete sqlite SQL query...                      │
    │   ### Sqlite SQL tables:                                │
    │   #                                                     │
    │   # table1 ( col1 , col2 )  ← Ranked                  │
    │   # table2 ( col3 , col4 )  ← by relevance             │
    │   # fk1 = fk2                                          │
    │   #                                                     │
    │   ### Question?                                         │
    │   SELECT"                                               │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Fine-tuned Model Inference                              │
    │                                                         │
    │  Input: Question + Ranked Schema                        │
    │         ↓                                               │
    │  Generate: "skeleton | actual_sql"                     │
    │         ↓                                               │
    │  Extract: actual_sql (after |)                         │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Final SQL Query                                         │
    │ "select count ( * ) from head where age > 56"          │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Execute & Evaluate                                      │
    │ - Execute on database                                   │
    │ - Compare with gold SQL                                 │
    │ - Calculate accuracy                                    │
    └─────────────────────────────────────────────────────────┘
```

## Detailed Component Flow

### 1. Data Preparation Flow

```
Raw Spider Data
    │
    ├─→ Extract SQL queries
    ├─→ Generate SQL skeletons
    ├─→ Format with instruction/response
    │
    └─→ train_sql_skeleton.jsonl
            │
            ├─→ Option A: Use directly (text format)
            │   └─→ Format: "### Instruction:\n...\n### Response:\n..."
            │
            └─→ Option B: Convert to OpenAI format
                └─→ train_sql_skeleton_openai.jsonl
                    └─→ Format: messages array
```

### 2. Schema Linking Flow (Inference)

```
Validation Question
    │
    ├─→ Load full database schema
    │   └─→ All tables, columns, foreign keys
    │
    ├─→ Schema Ranking Process
    │   ├─→ Rank tables by relevance
    │   │   └─→ Match question keywords
    │   │   └─→ Consider foreign key relationships
    │   │
    │   ├─→ Rank columns by relevance
    │   │   └─→ Match question keywords
    │   │   └─→ Consider data types
    │   │
    │   └─→ Remove irrelevant tables/columns
    │
    └─→ Ranked Schema
        └─→ Format: Clear context format
            └─→ "# table ( col1 , col2 )"
```

### 3. Training Flow

```
Base Model (Qwen 2.5 7B Instruct)
    │
    ├─→ Load with LoRA/QLoRA
    │   └─→ Quantization (if QLoRA)
    │
    ├─→ Training Data
    │   └─→ train_sql_skeleton_openai.jsonl
    │       └─→ Format: messages with skeleton | sql
    │
    ├─→ Training Process
    │   ├─→ Tokenize full sequence
    │   ├─→ Mask system/user messages
    │   ├─→ Compute loss on assistant messages
    │   └─→ Update LoRA weights
    │
    └─→ Fine-tuned Adapter
        └─→ ./qwen-skeleton-qlora-adapter
```

### 4. Inference Flow

```
User Question
    │
    ├─→ Schema Linking
    │   └─→ Generate ranked schema
    │
    ├─→ Format Prompt
    │   └─→ Question + Ranked Schema
    │
    ├─→ Model Inference
    │   └─→ Generate: "skeleton | actual_sql"
    │
    ├─→ Post-processing
    │   └─→ Extract actual_sql (after |)
    │
    └─→ Execute SQL
        └─→ Get results
```

## Key Components

### Data Format

**Training Data:**
- **Skeleton format**: `select count ( _ ) from _ where _ | select count ( * ) from head where age > 56`
- **Purpose**: Reduces complexity, separates structure from content
- **Benefit**: +10-11% accuracy improvement

### Schema Linking

**Process:**
1. **Table Ranking**: Order tables by relevance to question
2. **Column Ranking**: Order columns within each table
3. **Filtering**: Remove irrelevant tables/columns
4. **Formatting**: Convert to clear context format

**Benefit**: +2.7% accuracy improvement

### Loss Masking

**During Training:**
- System messages: Masked (no loss)
- User messages: Masked (no loss)
- Assistant messages: Loss computed (only these tokens)

**Result**: Model learns to generate responses, not instructions

## Pipeline Commands

### 1. Data Preparation
```bash
# Generate skeleton training data
python generate-finetune-data.py \
    --mode train \
    --sql_type sql \
    --skeleton \
    --clear_context

# Convert to OpenAI format (optional)
python train/convert_to_openai_format.py
```

### 2. Schema Ranking (for validation)
```bash
# Generate ranked schema for validation set
python schema_linking.py
# or
python schema_link_redo.py
```

### 3. Training
```bash
# Using LlamaFactory
llamafactory-cli train train/qwen_skeleton_llamafactory.yaml

# Or using Transformers+PEFT+TRL
python train/qwen_lora_finetune.py \
    --train_data data/train_sql_skeleton_openai.jsonl \
    --output_dir ./qwen-skeleton-qlora-adapter
```

### 4. Inference with Schema Linking
```bash
# Generate predictions with ranked schema
python eval/gen_predictions_qwen.py \
    --model_path ./qwen-skeleton-qlora-adapter \
    --input_file data/validation_sql_ranked.json \
    --output_file predictions/qwen_skeleton.txt
```

## Performance Improvements

| Component | Improvement | Cumulative |
|-----------|-------------|------------|
| Base SQL | 50% | 50% |
| + Skeleton Format | +10-11% | 60-61% |
| + Schema Ranking | +2.7% | 63-64% |
| + Clear Context | +1-2% | 64-65% |

## File Structure

```
text-to-sql-wizardcoder/
├── data/
│   ├── train_sql_skeleton.jsonl          # Original skeleton format
│   ├── train_sql_skeleton_openai.jsonl   # OpenAI format (for training)
│   └── validation_sql_ranked.json         # Ranked schema (for inference)
├── train/
│   ├── qwen_skeleton_llamafactory.yaml    # LlamaFactory config
│   ├── dataset_info.json                  # Dataset configuration
│   ├── qwen_lora_finetune.py              # Training script
│   └── convert_to_openai_format.py        # Data converter
└── eval/
    ├── gen_predictions_qwen.py            # Inference script
    └── evaluation.py                       # Evaluation script
```

## Summary

**Training Phase:**
- Uses skeleton data format
- Trains model to generate: `skeleton | actual_sql`
- Loss computed only on assistant messages

**Inference Phase:**
- Applies schema linking before inference
- Uses ranked schema (relevant tables/columns first)
- Model generates skeleton | sql format
- Extracts actual SQL for execution

This pipeline combines the benefits of skeleton format (+10-11%) with schema ranking (+2.7%) for optimal text-to-SQL performance.


