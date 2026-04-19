"""
Example script for generating NatSQL predictions instead of SQL.
This is a modified version of gen_predictions_chatgpt.py that generates NatSQL.

Usage:
    python gen_predictions_chatgpt_natsql.py --output test/natsql_chatgpt.txt
"""

from datasets import load_dataset
from tqdm import tqdm
from scripts.helpers import chatgpt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os

def format(response_text):
    response_text = response_text.strip().replace("\n", " ").replace("\t", " ")
    if not response_text.upper().startswith('SELECT'):
        response_text = 'SELECT ' + response_text
    return response_text

def fetch_response(i, entry, args):
    # Modified prompt to generate NatSQL instead of SQL
    prompt = f"""
    ### Complete sqlite NatSQL query only and with no explanation, and do not select extra columns that are not explicitly requested in the query.
    ### NatSQL is a simplified SQL representation that makes joins easier. Use table.* for aggregations, and join conditions in WHERE clause.
    ### Sqlite SQL tables, with their properties:
    #
    {entry['db_info']}
    #
    ### {entry['question']}
    SELECT
    """

    # Define messages to keep track of the message history
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Generate NatSQL prediction
    response_text = format(chatgpt(messages, model='gpt-3.5-turbo'))
    
    return i, response_text

def main(args):
    # Load validation dataset
    with open('../data/validation_sql_clear.json', 'r') as f:
        dataset = json.load(f)

    responses = {}
    # Using ThreadPoolExecutor to parallelize the work
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(fetch_response, i, entry, args) for i, entry in enumerate(dataset)]
        for future in tqdm(as_completed(futures), total=len(dataset), desc="Generating NatSQL responses"):
            idx, response_text = future.result()
            responses[idx] = response_text

    # Sort responses by index and then write to the file
    sorted_responses = [responses[i] for i in sorted(responses.keys())]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write NatSQL predictions
    with open(args.output, 'w') as f:
        for response_text in sorted_responses:
            f.write(response_text + "\n")
    
    print(f"\n✓ Generated {len(sorted_responses)} NatSQL predictions")
    print(f"✓ Saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Convert NatSQL to SQL:")
    print(f"   python convert_natsql_to_sql.py --input_file {args.output} --output_file {args.output.replace('.txt', '_converted.txt')}")
    print(f"2. Evaluate the converted SQL:")
    print(f"   python evaluation.py --input {args.output.replace('.txt', '_converted.txt')} --natsql")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NatSQL queries using ChatGPT.')
    parser.add_argument('--output', type=str, default='test/natsql_chatgpt.txt', help="Output file path for NatSQL predictions.")
    parser.add_argument('--threads', type=int, default=8, help="Number of threads to use.")
    args = parser.parse_args()
    main(args)










