"""
Convert train_sql_skeleton.jsonl to format that LlamaFactory expects for alpaca format.

LlamaFactory's alpaca format expects separate instruction and response fields,
but our data has everything in a single "text" field with markers.
"""

import json
import re

def convert_to_alpaca_format(input_file, output_file):
    """Convert combined text format to separate instruction/response format"""
    
    converted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            
            # Split on ### Response: marker
            if '### Response:' in text:
                parts = text.split('### Response:')
                instruction_part = parts[0]
                response_part = parts[1].strip() if len(parts) > 1 else ''
                
                # Remove ### Instruction: marker and clean up
                instruction = instruction_part.replace('### Instruction:', '').strip()
                instruction = instruction.replace('Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.', '').strip()
                
                # Create alpaca format
                converted_entry = {
                    "instruction": instruction,
                    "input": "",  # Empty if no separate input
                    "output": response_part
                }
                
                converted_data.append(converted_entry)
            else:
                print(f"Warning: No ### Response: marker found in entry")
    
    # Write converted data
    with open(output_file, 'w') as f:
        for entry in converted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} examples")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    convert_to_alpaca_format(
        input_file=os.path.join(project_root, "data/train_sql_skeleton.jsonl"),
        output_file=os.path.join(project_root, "data/train_sql_skeleton_alpaca.jsonl")
    )

