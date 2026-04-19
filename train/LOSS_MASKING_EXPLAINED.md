# How SFTTrainer Masks Instruction Tokens and Computes Loss Only on Response

## Overview

`SFTTrainer` from the `trl` library automatically identifies instruction vs response tokens and masks instruction tokens so loss is only computed on response tokens.

## How It Works

### 1. **Automatic Detection of Instruction/Response Boundaries**

SFTTrainer looks for special markers in your text:
- `### Instruction:` or `### Instruction`
- `### Response:` or `### Response`

Your data format:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Convert text to sql: How many heads... | schema...

### Response:

select count ( _ ) from _ where _ | select count ( * ) from head where age > 56
```

### 2. **Label Creation Process**

Here's what happens internally:

```python
# Step 1: Tokenize the full sequence
full_text = "### Instruction:\n...\n### Response:\nselect count..."
tokens = tokenizer(full_text)  # [token_0, token_1, ..., token_N]

# Step 2: Identify instruction vs response tokens
instruction_end = find_marker_position(tokens, "### Response:")
# instruction_end = position 150 (example)

# Step 3: Create labels
labels = []
for i, token in enumerate(tokens):
    if i < instruction_end:
        labels.append(-100)  # Mask instruction tokens
    else:
        labels.append(token_id)  # Use actual token for response

# Result:
# labels = [-100, -100, ..., -100, token_150, token_151, ..., token_N]
#          ↑ instruction (masked)  ↑ response (loss computed)
```

### 3. **Loss Computation**

During training:

```python
# Forward pass
logits = model(input_ids)  # Shape: [batch, seq_len, vocab_size]

# Loss computation (simplified)
loss_per_token = cross_entropy_loss(logits, labels)

# Masked positions (labels = -100) are ignored
masked_loss = loss_per_token * (labels != -100)

# Final loss = mean of non-masked tokens only
total_loss = masked_loss.sum() / (labels != -100).sum()
```

## Visual Example

```
Input Sequence:
┌─────────────────────────────────────────────────────────┐
│ "### Instruction:\nConvert text to sql: ...\n           │
│  ### Response:\nselect count(*) from head"              │
└─────────────────────────────────────────────────────────┘
         ↓ Tokenize
┌─────────────────────────────────────────────────────────┐
│ Tokens: [1234, 5678, ..., 9012, 3456, 7890, ...]       │
│         ↑ instruction tokens    ↑ response tokens      │
└─────────────────────────────────────────────────────────┘
         ↓ Create Labels
┌─────────────────────────────────────────────────────────┐
│ Labels: [-100, -100, ..., -100, 3456, 7890, ...]      │
│         ↑ masked (no loss)    ↑ loss computed          │
└─────────────────────────────────────────────────────────┘
         ↓ Compute Loss
┌─────────────────────────────────────────────────────────┐
│ Loss:   [0, 0, ..., 0, 0.5, 0.3, ...]                  │
│         ↑ ignored          ↑ contributes to loss        │
└─────────────────────────────────────────────────────────┘
```

## Code Implementation in SFTTrainer

SFTTrainer does this automatically. Here's the relevant part:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",  # Field containing full instruction+response
    max_seq_length=2048,
    tokenizer=tokenizer,
    # SFTTrainer automatically:
    # 1. Finds "### Instruction:" and "### Response:" markers
    # 2. Masks instruction tokens (sets labels to -100)
    # 3. Computes loss only on response tokens
)
```

## Key Points

### ✅ What SFTTrainer Does Automatically:

1. **Detects markers**: Looks for `### Instruction:` and `### Response:`
2. **Creates labels**: Sets instruction tokens to `-100` (ignored in loss)
3. **Computes loss**: Only on response tokens
4. **Handles padding**: Also masks padding tokens

### 📝 Your Data Format:

Your data already has the correct format:
```json
{
  "text": "### Instruction:\n...\n### Response:\nselect count..."
}
```

This is exactly what SFTTrainer expects!

## Manual Verification (Optional)

If you want to see how labels are created, you can check:

```python
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Create collator that handles instruction/response masking
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    instruction_template="### Instruction:",
    response_template="### Response:",
    ignore_index=-100,
)

# Example text
example = {
    "text": "### Instruction:\nConvert text to sql: How many heads?\n### Response:\nselect count(*) from head"
}

# Tokenize
tokenized = tokenizer(example["text"], return_tensors="pt")

# Create labels (this is what SFTTrainer does internally)
labels = collator([example])["labels"]

# Check which tokens are masked
print("Labels:", labels)
# Instruction tokens will be -100 (masked)
# Response tokens will be actual token IDs
```

## Summary

**SFTTrainer automatically:**
1. ✅ Detects `### Instruction:` and `### Response:` markers
2. ✅ Masks instruction tokens (labels = -100)
3. ✅ Computes loss only on response tokens
4. ✅ You don't need to do anything special!

**Your data format is correct** - just make sure you have:
- `### Instruction:` marker
- `### Response:` marker
- Full text in the `text` field

The trainer handles everything else automatically! 🎉




