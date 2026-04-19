# LlamaFactory Columns Configuration Explained

## Why Some Fields Are Empty Strings ("")

In your `dataset_info.json`, you have:

```json
"columns": {
  "prompt": "text",
  "query": "",
  "response": "",
  "history": ""
}
```

### What Each Field Means

| Field | Purpose | Why Empty? |
|-------|---------|------------|
| **`prompt`** | Main instruction/input field | ✅ Set to `"text"` - points to your data's `text` field |
| **`query`** | Separate query field (optional) | ❌ Empty - not used in your format |
| **`response`** | Separate response field (optional) | ❌ Empty - not used in your format |
| **`history`** | Conversation history (optional) | ❌ Empty - not used in your format |

### Your Data Format

Your data has everything in a **single `text` field**:

```json
{
  "db_id": "...",
  "text": "### Instruction:\n...\n### Response:\nselect count(*)..."
}
```

Since instruction and response are combined in one field, you only need:
- `"prompt": "text"` - tells LlamaFactory to use the `text` field

The other fields (`query`, `response`, `history`) are empty because:
1. **You don't have separate fields** - everything is in `text`
2. **They're optional** - only needed for other data formats
3. **LlamaFactory ignores empty strings** - they're just placeholders

## Different Data Formats Use Different Fields

### Format 1: Your Current Format (Alpaca-style)
```json
{
  "text": "### Instruction:\n...\n### Response:\n..."
}
```
**Columns needed:**
- `prompt: "text"` ✅
- `query: ""` ❌ (not used)
- `response: ""` ❌ (not used)
- `history: ""` ❌ (not used)

### Format 2: Separate Fields Format
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```
**Columns needed:**
- `prompt: "instruction"` ✅
- `query: "input"` ✅
- `response: "output"` ✅
- `history: ""` ❌ (not used)

### Format 3: Conversation History Format
```json
{
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "history": [...]
}
```
**Columns needed:**
- `prompt: "conversation"` ✅
- `query: ""` ❌
- `response: ""` ❌
- `history: "history"` ✅

## Simplified Configuration

Since you only use the `text` field, you can simplify to:

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

The empty fields (`query`, `response`, `history`) are optional and can be omitted or left as empty strings - LlamaFactory will ignore them.

## Summary

- ✅ **`prompt: "text"`** - Required, points to your data's `text` field
- ❌ **`query: ""`** - Empty = not used (your format doesn't have separate query field)
- ❌ **`response: ""`** - Empty = not used (response is inside `text` field)
- ❌ **`history: ""`** - Empty = not used (no conversation history)

**Empty strings = "This field is not used in my data format"**

This is correct for your data! ✅




