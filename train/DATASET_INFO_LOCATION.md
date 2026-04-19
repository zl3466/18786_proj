# How LlamaFactory Finds dataset_info.json

## The Key: `dataset_dir` Parameter

In your YAML config, you have:

```yaml
### dataset
dataset: sql_skeleton
dataset_dir: train
```

## How It Works

LlamaFactory uses `dataset_dir` to locate `dataset_info.json`:

1. **Reads `dataset_dir`** from YAML: `dataset_dir: train`
2. **Looks for `dataset_info.json`** in that directory: `train/dataset_info.json`
3. **Finds your dataset definition** by name: `sql_skeleton`

## File Structure

```
text-to-sql-wizardcoder/
├── train/
│   ├── dataset_info.json          ← LlamaFactory looks here
│   └── qwen_skeleton_llamafactory.yaml
└── data/
    └── train_sql_skeleton.jsonl
```

## Step-by-Step Process

1. **You run:**
   ```bash
   llamafactory-cli train train/qwen_skeleton_llamafactory.yaml
   ```

2. **LlamaFactory reads YAML:**
   ```yaml
   dataset_dir: train
   dataset: sql_skeleton
   ```

3. **LlamaFactory looks for:**
   ```
   {dataset_dir}/dataset_info.json
   → train/dataset_info.json
   ```

4. **LlamaFactory finds:**
   ```json
   {
     "sql_skeleton": {
       "file_name": "../data/train_sql_skeleton.jsonl",
       ...
     }
   }
   ```

5. **LlamaFactory uses:**
   - Dataset name: `sql_skeleton`
   - File path: `../data/train_sql_skeleton.jsonl` (relative to `train/` directory)

## Path Resolution

All paths in `dataset_info.json` are **relative to `dataset_dir`**:

```json
{
  "sql_skeleton": {
    "file_name": "../data/train_sql_skeleton.jsonl"
    // This path is relative to train/ directory
    // So it resolves to: train/../data/train_sql_skeleton.jsonl
    // Which is: data/train_sql_skeleton.jsonl
  }
}
```

## Examples

### Example 1: Current Setup
```yaml
dataset_dir: train
```
- Looks for: `train/dataset_info.json` ✅
- Your file: `train/dataset_info.json` ✅
- **Works!**

### Example 2: Different Directory
```yaml
dataset_dir: configs
```
- Looks for: `configs/dataset_info.json`
- Your file: `train/dataset_info.json`
- **Would fail!** (unless you move the file)

### Example 3: Root Directory
```yaml
dataset_dir: .
```
- Looks for: `./dataset_info.json` (root directory)
- Your file: `train/dataset_info.json`
- **Would fail!** (unless you move the file)

## Summary

| YAML Setting | Looks For | Your File | Result |
|--------------|-----------|-----------|--------|
| `dataset_dir: train` | `train/dataset_info.json` | `train/dataset_info.json` | ✅ Works |
| `dataset_dir: .` | `./dataset_info.json` | `train/dataset_info.json` | ❌ Fails |
| `dataset_dir: configs` | `configs/dataset_info.json` | `train/dataset_info.json` | ❌ Fails |

## Your Current Setup is Correct ✅

```yaml
dataset_dir: train  # Looks in train/ directory
```

```json
// train/dataset_info.json
{
  "sql_skeleton": {
    "file_name": "../data/train_sql_skeleton.jsonl"
  }
}
```

**LlamaFactory will:**
1. Find `train/dataset_info.json` ✅
2. Look up `sql_skeleton` dataset ✅
3. Load data from `../data/train_sql_skeleton.jsonl` ✅

Everything is correctly configured! 🎉




