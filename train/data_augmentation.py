"""
Data Augmentation Script for Text-to-SQL Training

This script applies various data augmentation techniques to improve
model generalization.
"""

import json
import argparse
import random
import re
from typing import List, Dict, Set
from collections import defaultdict

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON or JSONL file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_dataset(data: List[Dict], file_path: str):
    """Save dataset to JSON or JSONL file"""
    if file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {file_path}")

def extract_schema_names(text: str) -> Dict[str, Set[str]]:
    """Extract table and column names from schema"""
    tables = set()
    columns = set()
    
    # Extract table names (patterns like "table_name :" or "# table_name (")
    table_pattern = r'(?:#\s*|)(\w+)\s*[:(]'
    tables.update(re.findall(table_pattern, text, re.IGNORECASE))
    
    # Extract column names (between parentheses or after colons)
    column_pattern = r'\(([^)]+)\)'
    for match in re.finditer(column_pattern, text):
        cols = [c.strip() for c in match.group(1).split(',')]
        columns.update(cols)
    
    return {'tables': tables, 'columns': columns}

def create_synonym_dict(schema_names: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Create synonym dictionary for schema names.
    In practice, you might want to use a more sophisticated approach
    or maintain a manual synonym dictionary.
    """
    synonyms = defaultdict(list)
    
    # Simple heuristic: create synonyms based on common patterns
    for name in schema_names['tables'] | schema_names['columns']:
        name_lower = name.lower()
        
        # Common synonyms
        synonym_map = {
            'id': ['identifier', 'pk'],
            'name': ['title', 'label'],
            'date': ['time', 'timestamp'],
            'count': ['number', 'amount'],
            'price': ['cost', 'amount'],
            'user': ['person', 'customer'],
            'order': ['purchase', 'transaction'],
        }
        
        for key, values in synonym_map.items():
            if key in name_lower:
                for val in values:
                    new_name = name_lower.replace(key, val)
                    if new_name != name_lower:
                        synonyms[name].append(new_name)
    
    return dict(synonyms)

def augment_with_schema_synonyms(entry: Dict, synonym_dict: Dict[str, List[str]], prob: float = 0.3) -> Dict:
    """
    Replace schema names with synonyms (only in question, not SQL)
    Note: This is a simplified version. In practice, you'd need to
    be careful not to break SQL syntax.
    """
    if random.random() > prob:
        return entry
    
    text = entry.get('text', '')
    question = entry.get('question', '')
    
    # Only augment question part, not SQL
    if '### Instruction:' in text:
        parts = text.split('### Response:')
        instruction_part = parts[0]
        response_part = parts[1] if len(parts) > 1 else ''
    else:
        instruction_part = text
        response_part = ''
    
    # Replace synonyms in instruction/question
    augmented_instruction = instruction_part
    for original, synonyms in synonym_dict.items():
        if synonyms and random.random() < 0.5:
            synonym = random.choice(synonyms)
            # Simple replacement (case-insensitive)
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            augmented_instruction = pattern.sub(synonym, augmented_instruction, count=1)
    
    # Reconstruct text
    if response_part:
        augmented_text = augmented_instruction + '### Response:\n\n' + response_part
    else:
        augmented_text = augmented_instruction
    
    new_entry = entry.copy()
    new_entry['text'] = augmented_text
    return new_entry

def paraphrase_question_simple(question: str) -> List[str]:
    """
    Simple question paraphrasing using patterns.
    For better results, use ChatGPT API or another LLM.
    """
    paraphrases = []
    
    # Pattern-based paraphrases
    patterns = [
        (r'How many (.+?) are there', r'What is the count of \1'),
        (r'What is the (.+?) of (.+?)', r'Show the \1 of \2'),
        (r'List (.+?)', r'Show all \1'),
        (r'Find (.+?)', r'Get \1'),
        (r'Show (.+?)', r'Display \1'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, question, re.IGNORECASE):
            paraphrased = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
            if paraphrased != question:
                paraphrases.append(paraphrased)
    
    return paraphrases

def augment_with_paraphrases(entry: Dict, num_paraphrases: int = 2) -> List[Dict]:
    """
    Create augmented entries with paraphrased questions.
    This keeps the same SQL but changes the question wording.
    """
    question = entry.get('question', '')
    if not question:
        return [entry]
    
    # Extract question from text if needed
    if '### Instruction:' in entry.get('text', ''):
        match = re.search(r'Convert text to \w+:\s*(.+?)(?:\s+\||\n)', entry['text'])
        if match:
            question = match.group(1).strip()
    
    paraphrases = paraphrase_question_simple(question)
    
    if not paraphrases:
        return [entry]
    
    augmented_entries = []
    for para in paraphrases[:num_paraphrases]:
        new_entry = entry.copy()
        new_text = entry['text'].replace(question, para)
        new_entry['text'] = new_text
        if 'question' in new_entry:
            new_entry['question'] = para
        augmented_entries.append(new_entry)
    
    return augmented_entries if augmented_entries else [entry]

def balance_by_complexity(dataset: List[Dict]) -> List[Dict]:
    """
    Balance dataset by SQL query complexity.
    Ensures equal representation of different query types.
    """
    def get_query_complexity(entry: Dict) -> str:
        text = entry.get('text', entry.get('ground_truth', ''))
        sql = text.split('|')[-1] if '|' in text else text
        sql_upper = sql.upper()
        
        if 'UNION' in sql_upper or 'INTERSECT' in sql_upper:
            return 'complex'
        elif 'JOIN' in sql_upper:
            return 'join'
        elif any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            return 'aggregation'
        else:
            return 'simple'
    
    # Categorize
    categories = defaultdict(list)
    for entry in dataset:
        cat = get_query_complexity(entry)
        categories[cat].append(entry)
    
    # Balance by sampling equally from each category
    min_size = min(len(cat_list) for cat_list in categories.values())
    balanced = []
    
    for cat, entries in categories.items():
        if len(entries) > min_size:
            sampled = random.sample(entries, min_size)
            balanced.extend(sampled)
        else:
            balanced.extend(entries)
    
    # Shuffle
    random.shuffle(balanced)
    
    return balanced

def apply_curriculum_learning(dataset: List[Dict]) -> List[Dict]:
    """Sort dataset by complexity for curriculum learning"""
    def get_complexity_score(entry: Dict) -> int:
        text = entry.get('text', entry.get('ground_truth', ''))
        sql = text.split('|')[-1] if '|' in text else text
        sql_upper = sql.upper()
        
        score = 0
        score += sql_upper.count('JOIN') * 2
        if 'GROUP BY' in sql_upper:
            score += 1
        if 'HAVING' in sql_upper:
            score += 1
        if 'UNION' in sql_upper or 'INTERSECT' in sql_upper:
            score += 2
        if sql_upper.count('SELECT') > 1:
            score += 2
        return score
    
    # Sort by complexity
    sorted_dataset = sorted(dataset, key=get_complexity_score)
    return sorted_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Augment text-to-SQL training data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file (JSON/JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output augmented dataset file"
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Add paraphrased questions"
    )
    parser.add_argument(
        "--synonyms",
        action="store_true",
        help="Apply schema synonym replacement"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance dataset by query complexity"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Sort by complexity (curriculum learning)"
    )
    parser.add_argument(
        "--augment_ratio",
        type=float,
        default=0.3,
        help="Ratio of examples to augment (0.0-1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from: {args.input}")
    dataset = load_dataset(args.input)
    print(f"Original dataset size: {len(dataset)}")
    
    augmented_dataset = []
    
    # Process each entry
    for entry in dataset:
        augmented_dataset.append(entry)  # Keep original
        
        # Apply augmentations based on probability
        if random.random() < args.augment_ratio:
            augmented_entry = entry.copy()
            
            # Apply synonym replacement
            if args.synonyms:
                schema_names = extract_schema_names(entry.get('text', ''))
                synonym_dict = create_synonym_dict(schema_names)
                augmented_entry = augment_with_schema_synonyms(augmented_entry, synonym_dict)
            
            # Apply paraphrasing
            if args.paraphrase:
                paraphrased = augment_with_paraphrases(entry, num_paraphrases=1)
                augmented_dataset.extend(paraphrased)
                continue  # Skip other augmentations if paraphrasing
            
            if augmented_entry != entry:
                augmented_dataset.append(augmented_entry)
    
    print(f"After augmentation: {len(augmented_dataset)} examples")
    
    # Apply balancing if requested
    if args.balance:
        print("Balancing dataset by complexity...")
        augmented_dataset = balance_by_complexity(augmented_dataset)
        print(f"After balancing: {len(augmented_dataset)} examples")
    
    # Apply curriculum learning if requested
    if args.curriculum:
        print("Applying curriculum learning (sorting by complexity)...")
        augmented_dataset = apply_curriculum_learning(augmented_dataset)
    
    # Save augmented dataset
    print(f"Saving augmented dataset to: {args.output}")
    save_dataset(augmented_dataset, args.output)
    
    print("\nData augmentation completed!")

if __name__ == "__main__":
    main()










