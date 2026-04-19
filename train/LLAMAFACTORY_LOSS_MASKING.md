# LlamaFactory Loss Masking: How It Ignores Instruction Tokens

## Yes, LlamaFactory Also Ignores Instruction Tokens! ✅

LlamaFactory automatically masks instruction tokens and computes loss only on response tokens, similar to SFTTrainer.

## How LlamaFactory Does It

### 1. **Template-Based Detection**

LlamaFactory uses **templates** to identify instruction/response boundaries. Your configuration:

```yaml
# In qwen_skeleton_llamafactory.yaml
template: default  # Uses Qwen's default template
```

The `default` template for Qwen knows about:
- `### Instruction:`
- `### Response:`

### 2. **Dataset Info Configuration**

Your `dataset_info.json` explicitly defines the tags:

```json
{
  "sql_skeleton": {
    "tags": {
      "instruction_tag": "### Instruction:",
      "response_tag": "### Response:",
      "user_tag": "",
      "assistant_tag": ""
    }
  }
}
```

This tells LlamaFactory:
- Where instruction starts: `### Instruction:`
- Where response starts: `### Response:`
- Everything before `### Response:` = instruction (masked)
- Everything after `### Response:` = response (loss computed)

### 3. **Internal Process**

LlamaFactory does this automatically:

```python
# Step 1: Parse your text
text = "### Instruction:\n...\n### Response:\nselect count(*)"

# Step 2: Find response start position
response_start = text.find("### Response:") + len("### Response:")

# Step 3: Create labels
labels = []
for i, token in enumerate(tokens):
    if token_position < response_start:
        labels.append(-100)  # Mask instruction
    else:
        labels.append(token_id)  # Compute loss on response
```

## Comparison: SFTTrainer vs LlamaFactory

| Feature | SFTTrainer | LlamaFactory |
|---------|-----------|--------------|
| **Automatic Detection** | ✅ Yes | ✅ Yes |
| **Marker Detection** | `### Instruction:` / `### Response:` | Same markers |
| **Label Masking** | Sets instruction tokens to -100 | Sets instruction tokens to -100 |
| **Configuration** | Automatic (no config needed) | Template + dataset_info.json |
| **Result** | Loss only on response | Loss only on response |

## Your Configuration is Correct ✅

Looking at your setup:

1. **Template**: `template: default` - Uses Qwen's default template
2. **Tags**: Defined in `dataset_info.json`:
   ```json
   "instruction_tag": "### Instruction:",
   "response_tag": "### Response:"
   ```
3. **Data Format**: Your data has these markers:
   ```
   ### Instruction:
   ...
   ### Response:
   select count(*)...
   ```

This is exactly what LlamaFactory needs!

## How to Verify

If you want to verify LlamaFactory is masking correctly, you can check the training logs. LlamaFactory will:

1. Parse your data using the template
2. Identify instruction/response boundaries
3. Create labels with instruction tokens masked
4. Compute loss only on response tokens

## Summary

**Yes, LlamaFactory ignores instruction tokens!**

- ✅ Uses template to detect `### Instruction:` and `### Response:`
- ✅ Masks instruction tokens (sets labels to -100)
- ✅ Computes loss only on response tokens
- ✅ Your configuration is correct

Both SFTTrainer and LlamaFactory handle this the same way - automatically masking instruction tokens and computing loss only on response tokens.




