# Data Format Comparison: train_sql_chatgpt.json vs train_sql_skeleton_openai.jsonl

## Format Comparison

### train_sql_chatgpt.json (Current - NO Skeleton)

```json
{
  "messages": [
    {"role": "system", "content": "You are an excellent SQL writer."},
    {"role": "user", "content": "### Complete sqlite SQL query...\nSELECT \n"},
    {"role": "assistant", "content": "count ( * ) from head where age > 56"}
  ]
}
```

**Issues:**
- ❌ **Missing skeleton format** - Only has actual SQL
- ❌ **Lower expected accuracy** - ~50% (baseline)
- ✅ Format is correct (messages format works with LlamaFactory)

### train_sql_skeleton_openai.jsonl (Recommended - WITH Skeleton)

```json
{
  "messages": [
    {"role": "system", "content": "You are an excellent SQL writer."},
    {"role": "user", "content": "### Complete sqlite SQL query...\nSELECT \n"},
    {"role": "assistant", "content": "select count ( _ ) from _ where _ | select count ( * ) from head where age > 56"}
  ]
}
```

**Advantages:**
- ✅ **Includes skeleton format** - `skeleton | actual_sql`
- ✅ **Higher expected accuracy** - ~60-61% (+10-11% improvement)
- ✅ Format is correct (messages format works with LlamaFactory)

## Key Difference

| Aspect | train_sql_chatgpt.json | train_sql_skeleton_openai.jsonl |
|--------|----------------------|--------------------------------|
| **Format** | ✅ Messages (correct) | ✅ Messages (correct) |
| **Skeleton** | ❌ Missing | ✅ Included |
| **Assistant Content** | `count ( * ) from head...` | `select count ( _ ) from _ where _ \| select count ( * ) from head...` |
| **Expected Accuracy** | ~50% | ~60-61% |
| **Improvement** | Baseline | +10-11% |

## Recommendation

### ❌ Don't Use train_sql_chatgpt.json

**Why:**
- Missing skeleton format (loses +10-11% accuracy)
- Lower baseline performance
- Doesn't leverage the key insight from your research

### ✅ Use train_sql_skeleton_openai.jsonl

**Why:**
- Includes skeleton format (+10-11% improvement)
- Same messages format (works with LlamaFactory)
- Matches your best-performing approach

## How to Use

### Option 1: Use Existing Skeleton Data

```yaml
# In qwen_skeleton_llamafactory.yaml
dataset: sql_skeleton  # Uses train_sql_skeleton_openai.jsonl
```

### Option 2: Convert train_sql_chatgpt.json to Include Skeleton

You would need to:
1. Add skeleton generation to the data
2. Format as: `skeleton | actual_sql`

But it's easier to just use the existing `train_sql_skeleton_openai.jsonl` which already has this.

## Summary

**train_sql_chatgpt.json:**
- ✅ Format is correct (will work)
- ❌ Missing skeleton (loses 10-11% accuracy)
- ⚠️ Not recommended for best results

**train_sql_skeleton_openai.jsonl:**
- ✅ Format is correct (will work)
- ✅ Includes skeleton (+10-11% accuracy)
- ✅ Recommended for best results

**Verdict:** Use `train_sql_skeleton_openai.jsonl` instead for +10-11% better accuracy!

