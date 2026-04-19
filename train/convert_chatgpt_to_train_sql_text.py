"""
Convert train_sql_chatgpt.json / .jsonl (OpenAI messages: system / user / assistant)
into train_sql.json / .jsonl style rows: {"db_id": "...", "text": "<instruction + response>"}.

User prompts must match the layout produced by convert_to_openai_format.py:
  ### Sqlite SQL tables, with their properties:
  #
  # table ( col1 , col2 , ... )
  # fk.col = fk.col
  #
  ### <question>
  SELECT
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

HEADER = (
    "Below is an instruction that describes a task, paired with an input that provides further "
    "context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n"
)


def openai_schema_lines_to_pipe(schema_block: str) -> str:
    """Turn '# name ( cols )' / '# a.b = c.d' lines into pipe-separated train_sql schema."""
    parts: List[str] = []
    for raw in schema_block.strip().split("\n"):
        line = raw.strip()
        if not line or line == "#":
            continue
        if not line.startswith("#"):
            continue
        body = line[1:].strip()
        m = re.match(r"^(\S+)\s+\(\s*(.+)\s*\)\s*$", body)
        if m:
            table, cols = m.group(1), m.group(2).strip()
            parts.append(f"{table} : {cols}")
        else:
            parts.append(body.strip())
    return " | ".join(parts)


def parse_openai_user(user: str) -> Tuple[str, str]:
    """
    Returns (question, pipe_schema) where pipe_schema has no leading/trailing ' | ' wrappers
    (caller builds 'Convert text to sql: {q} | {pipe} | ').
    """
    m = re.search(
        r"### Sqlite SQL tables, with their properties:\n#\n(.*?)\n#\n### ",
        user,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not find Sqlite schema block in user message.")
    schema_block = m.group(1)
    tail = user[m.end() :]
    qm = re.match(r"([^\n]+)\nSELECT\s*$", tail.strip(), flags=re.IGNORECASE)
    if not qm:
        raise ValueError("Could not find question line before trailing SELECT.")
    question = qm.group(1).strip()
    pipe = openai_schema_lines_to_pipe(schema_block)
    return question, pipe


def assistant_to_response_sql(assistant: str) -> str:
    """Match train_sql.json: full SQL in ### Response, usually starting with 'select '."""
    s = assistant.strip()
    if not s:
        return s
    if re.match(r"(?i)^select\s", s):
        return s
    if " | " in s:
        return s
    return "select " + s


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    by_role = {m["role"]: m.get("content", "") for m in messages}
    user = by_role.get("user", "")
    assistant = by_role.get("assistant", "")
    question, pipe = parse_openai_user(user)
    instruction = f"Convert text to sql: {question} | {pipe} | "
    response = assistant_to_response_sql(assistant)
    return HEADER + instruction + "\n\n### Response:\n\n" + response


def load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array for .json input.")
    return data


def convert_record(item: Dict[str, Any]) -> Dict[str, str]:
    db_id = item.get("db_id", "") or ""
    msgs = item.get("messages")
    if not msgs:
        raise ValueError("Record missing 'messages'.")
    text = messages_to_text(msgs)
    return {"db_id": db_id, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ChatGPT messages dataset to train_sql-style text field format."
    )
    parser.add_argument(
        "input",
        help="Input .json (array) or .jsonl (one object per line) with 'messages'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (.json array or .jsonl). Default: input basename + _as_text + same ext.",
    )
    args = parser.parse_args()

    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_as_text{ext or '.jsonl'}"

    records = load_records(in_path)
    converted: List[Dict[str, str]] = []
    errors = 0
    for i, item in enumerate(records):
        try:
            converted.append(convert_record(item))
        except Exception as e:
            errors += 1
            print(f"Skip record {i}: {e}")

    if in_path.endswith(".jsonl") or out_path.endswith(".jsonl"):
        with open(out_path, "w", encoding="utf-8") as f:
            for row in converted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"Wrote {len(converted)} rows to {out_path}" + (f" ({errors} skipped)" if errors else ""))


if __name__ == "__main__":
    main()
