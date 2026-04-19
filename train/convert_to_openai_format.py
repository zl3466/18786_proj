"""
Convert train_sql_skeleton.jsonl to OpenAI messages format.

OpenAI format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import json
import os
import re

def extract_question_and_schema(instruction_text):
    """Extract question and schema from instruction text"""
    # Format: "Convert text to sql: {question} | {schema}"
    if "Convert text to sql:" in instruction_text:
        parts = instruction_text.split("Convert text to sql:", 1)
        if len(parts) > 1:
            rest = parts[1].strip()
            # Split on the first " | " to separate question from schema
            if " | " in rest:
                question, schema = rest.split(" | ", 1)
                return question.strip(), schema.strip()
            else:
                return rest, ""
    return instruction_text, ""

def format_schema_for_openai(schema_text):
    """Convert pipe-separated schema to OpenAI format"""
    if not schema_text:
        return ""
    
    # Schema format: "table1 : col1 , col2 | table2 : col3 , col4 | fk1 = fk2"
    lines = []
    
    # Split by " | " to get table definitions and foreign keys
    parts = schema_text.split(" | ")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if it's a foreign key (contains "=")
        if "=" in part and "." in part:
            lines.append(f"# {part}")
        # Check if it's a table definition (contains ":")
        elif ":" in part:
            table_name, columns = part.split(":", 1)
            table_name = table_name.strip()
            columns = columns.strip()
            # Format: "# table_name ( col1 , col2 , col3 )"
            lines.append(f"# {table_name} ( {columns} )")
    
    return "\n".join(lines)

def convert_to_openai_format(input_file, output_file):
    """Convert skeleton format to OpenAI messages format"""
    
    converted_data = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Split on ### Response: marker
                if '### Response:' in text:
                    parts = text.split('### Response:')
                    instruction_part = parts[0]
                    response_part = parts[1].strip() if len(parts) > 1 else ''
                    
                    # Extract instruction (remove header text)
                    instruction = instruction_part.replace('### Instruction:', '').strip()
                    instruction = instruction.replace(
                        'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.',
                        ''
                    ).strip()
                    
                    # Extract question and schema
                    question, schema = extract_question_and_schema(instruction)
                    
                    # Format schema for OpenAI
                    formatted_schema = format_schema_for_openai(schema)
                    
                    # Create user prompt (similar to train_sql_chatgpt.jsonl format)
                    # Format schema properly (remove trailing pipes, ensure proper formatting)
                    formatted_schema = formatted_schema.rstrip(" |").strip()
                    
                    user_prompt = f"""### Complete sqlite SQL query only and with no explanation, and do not select extra columns that are not explicitly requested in the query.
### Sqlite SQL tables, with their properties:
#
{formatted_schema}
#
### {question}
SELECT 
"""
                    # Clean up: remove trailing pipes from schema lines, but keep the structure
                    user_prompt = user_prompt.replace(" |\n", "\n")
                    
                    # Extract skeleton and actual SQL from response
                    # Format: "skeleton | actual_sql"
                    # Keep both skeleton and actual SQL (skeleton format)
                    if " | " in response_part:
                        skeleton, actual_sql = response_part.split(" | ", 1)
                        skeleton = skeleton.strip()
                        actual_sql = actual_sql.strip()
                        # Combine skeleton and actual SQL (keep full format)
                        assistant_content = f"{skeleton} | {actual_sql}"
                    else:
                        assistant_content = response_part.strip()
                    
                    # Note: We keep the full skeleton format, including "select" at the beginning
                    # This is different from train_sql_chatgpt.jsonl which removes "SELECT"
                    
                    # Create OpenAI format
                    converted_entry = {
                        "messages": [
                            {"role": "system", "content": "You are an excellent SQL writer."},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    
                    converted_data.append(converted_entry)
                else:
                    print(f"Warning: Line {line_num} - No ### Response: marker found")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Write converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} examples")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    convert_to_openai_format(
        input_file=os.path.join(project_root, "data/train_sql_skeleton.jsonl"),
        output_file=os.path.join(project_root, "data/train_sql_skeleton_openai.jsonl")
    )

