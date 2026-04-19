# LlamaFactory Dataset Format Guide

## Two Different Data Formats

You have two different data formats in your project:

### Format 1: Text Format (train_sql_skeleton.jsonl)
```json
{
  "db_id": "...",
  "text": "### Instruction:\n...\n### Response:\nselect count(*)..."
}
```

### Format 2: Messages Format (train_sql_chatgpt.jsonl)
```json
{
  "messages": [
    {"role": "system", "content": "You are an excellent SQL writer."},
    {"role": "user", "content": "### Complete sqlite SQL query..."},
    {"role": "assistant", "content": "count ( * ) from head..."}
  ]
}
```

## Configuration for Each Format

### For Text Format (train_sql_skeleton.jsonl)

```json
{
  "sql_skeleton": {
    "file_name": "../data/train_sql_skeleton.jsonl",
    "columns": {
      "prompt": "text"
    },
    "formatting": "alpaca",
    "tags": {
      "instruction_tag": "### Instruction:",
      "response_tag": "### Response:"
    }
  }
}
```

**Key points:**
- `formatting: "alpaca"` - For instruction-response text format
- `columns.prompt: "text"` - Points to the `text` field
- `tags` - Defines where instruction/response boundaries are

### For Messages Format (train_sql_chatgpt.jsonl)

```json
{
  "sql_chatgpt": {
    "file_name": "../data/train_sql_chatgpt.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

**Key points:**
- `formatting: "sharegpt"` - For messages array format
- `columns.messages: "messages"` - Points to the `messages` field
- `tags` - Defines role names (user, assistant, system)

## How to Use in Training Config

### Using Text Format:

```yaml
# In qwen_skeleton_llamafactory.yaml
dataset: sql_skeleton
```

### Using Messages Format:

```yaml
# In qwen_chatgpt_llamafactory.yaml
dataset: sql_chatgpt
```

## Differences

| Aspect | Text Format | Messages Format |
|--------|------------|-----------------|
| **Field Name** | `text` | `messages` |
| **Formatting** | `alpaca` | `sharegpt` |
| **Structure** | Single text string | Array of message objects |
| **Markers** | `### Instruction:` / `### Response:` | `role: "user"` / `role: "assistant"` |
| **System Message** | In text | Separate message with `role: "system"` |

## Which One to Use?

### Use Text Format (`sql_skeleton`) if:
- ✅ Your data has `text` field with `### Instruction:` / `### Response:`
- ✅ You want simpler structure
- ✅ You're using instruction-response format

### Use Messages Format (`sql_chatgpt`) if:
- ✅ Your data has `messages` array
- ✅ You want to include system messages
- ✅ You're using ChatGPT-style format

## Current Setup

Your `dataset_info.json` now supports **both formats**:

1. **sql_skeleton** - For `train_sql_skeleton.jsonl` (text format)
2. **sql_chatgpt** - For `train_sql_chatgpt.jsonl` (messages format)

You can use either one in your training config by changing:
```yaml
dataset: sql_skeleton  # or sql_chatgpt
```

## Loss Masking

Both formats automatically mask instruction/user tokens:
- **Text format**: Masks everything before `### Response:`
- **Messages format**: Masks system and user messages, computes loss on assistant messages

Both work correctly! ✅




