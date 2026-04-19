"""
Hard Example Mining Script

This script identifies hard examples from validation failures and adds them
to the training dataset for improved accuracy.
"""

import json
import argparse
from typing import List, Dict
from collections import defaultdict

def load_predictions(pred_file: str) -> Dict[str, str]:
    """Load predictions from file"""
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Assuming format: index|prediction or just prediction per line
                if '|' in line:
                    idx, pred = line.split('|', 1)
                    predictions[int(idx.strip())] = pred.strip()
                else:
                    # If no index, use line number
                    predictions[len(predictions)] = line
    return predictions

def load_gold_standard(gold_file: str) -> List[str]:
    """Load gold standard SQL queries"""
    gold = []
    with open(gold_file, 'r') as f:
        for line in f:
            gold.append(line.strip())
    return gold

def load_validation_data(val_file: str) -> List[Dict]:
    """Load validation dataset"""
    if val_file.endswith('.json'):
        with open(val_file, 'r') as f:
            return json.load(f)
    elif val_file.endswith('.jsonl'):
        data = []
        with open(val_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file format: {val_file}")

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    import re
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    # Lowercase for comparison (optional)
    # sql = sql.lower()
    return sql

def identify_hard_examples(
    validation_data: List[Dict],
    predictions: Dict[int, str],
    gold_standard: List[str],
    use_execution: bool = False,
    db_dir: str = None,
) -> List[Dict]:
    """
    Identify hard examples (incorrect predictions)
    
    Args:
        validation_data: List of validation examples
        predictions: Dictionary mapping index to prediction
        gold_standard: List of gold standard SQL queries
        use_execution: Whether to use execution accuracy (requires database)
        db_dir: Directory containing databases (for execution)
    
    Returns:
        List of hard examples
    """
    hard_examples = []
    
    for i, entry in enumerate(validation_data):
        if i not in predictions:
            continue
        
        pred = predictions[i]
        gold = gold_standard[i] if i < len(gold_standard) else None
        
        if gold is None:
            continue
        
        # Check if prediction matches gold
        is_correct = False
        
        if use_execution and db_dir:
            # Use execution-based comparison
            is_correct = check_execution_accuracy(
                pred, gold, entry.get('db_id'), db_dir
            )
        else:
            # Use string-based comparison (normalized)
            is_correct = normalize_sql(pred) == normalize_sql(gold)
        
        if not is_correct:
            hard_examples.append({
                'entry': entry,
                'prediction': pred,
                'gold': gold,
                'index': i,
            })
    
    return hard_examples

def check_execution_accuracy(
    pred_sql: str,
    gold_sql: str,
    db_id: str,
    db_dir: str,
) -> bool:
    """Check if prediction and gold SQL produce same results"""
    try:
        # Import evaluation utilities
        import sys
        sys.path.append('eval')
        from scripts.exec_eval import ExecEvaluator
        
        evaluator = ExecEvaluator(db_dir)
        result = evaluator.eval(pred_sql, gold_sql, db_id)
        return result['exact'] or result['exec']
    except Exception as e:
        print(f"Error checking execution accuracy: {e}")
        # Fall back to string comparison
        return normalize_sql(pred_sql) == normalize_sql(gold_sql)

def categorize_errors(hard_examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize hard examples by error type"""
    categories = defaultdict(list)
    
    for example in hard_examples:
        pred = example['prediction'].upper()
        gold = example['gold'].upper()
        
        error_type = "other"
        
        # Check for specific error patterns
        if 'JOIN' in gold and 'JOIN' not in pred:
            error_type = "missing_join"
        elif 'GROUP BY' in gold and 'GROUP BY' not in pred:
            error_type = "missing_groupby"
        elif 'HAVING' in gold and 'HAVING' not in pred:
            error_type = "missing_having"
        elif 'UNION' in gold or 'INTERSECT' in gold:
            error_type = "complex_query"
        elif 'COUNT' in gold or 'SUM' in gold or 'AVG' in gold:
            if 'COUNT' not in pred and 'SUM' not in pred and 'AVG' not in pred:
                error_type = "missing_aggregation"
        elif len(set(pred.split()) & set(gold.split())) / max(len(set(gold.split())), 1) < 0.5:
            error_type = "completely_wrong"
        else:
            error_type = "minor_error"
        
        categories[error_type].append(example)
    
    return dict(categories)

def add_to_training_data(
    hard_examples: List[Dict],
    train_file: str,
    output_file: str,
    oversample_factor: int = 3,
    max_additions: int = None,
):
    """
    Add hard examples to training data
    
    Args:
        hard_examples: List of hard examples to add
        train_file: Path to existing training data
        output_file: Path to output augmented training data
        oversample_factor: How many times to add each hard example
        max_additions: Maximum number of examples to add (None = all)
    """
    # Load existing training data
    print(f"Loading training data from: {train_file}")
    if train_file.endswith('.json'):
        with open(train_file, 'r') as f:
            train_data = json.load(f)
    elif train_file.endswith('.jsonl'):
        train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {train_file}")
    
    # Limit additions if specified
    if max_additions:
        hard_examples = hard_examples[:max_additions]
    
    # Convert hard examples to training format
    new_examples = []
    for example in hard_examples:
        entry = example['entry']
        # Add multiple copies (oversampling)
        for _ in range(oversample_factor):
            new_examples.append(entry)
    
    # Combine with existing data
    augmented_data = train_data + new_examples
    
    print(f"Original training examples: {len(train_data)}")
    print(f"Hard examples added: {len(new_examples)}")
    print(f"Total augmented examples: {len(augmented_data)}")
    
    # Save augmented dataset
    print(f"Saving augmented dataset to: {output_file}")
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    elif output_file.endswith('.jsonl'):
        with open(output_file, 'w') as f:
            for entry in augmented_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {output_file}")

def create_error_report(hard_examples: List[Dict], output_file: str):
    """Create a report of error types"""
    categories = categorize_errors(hard_examples)
    
    report = []
    report.append("=" * 80)
    report.append("HARD EXAMPLE MINING REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal hard examples: {len(hard_examples)}\n")
    report.append("Error Categories:\n")
    
    for error_type, examples in sorted(categories.items(), key=lambda x: -len(x[1])):
        report.append(f"  {error_type}: {len(examples)} examples ({len(examples)/len(hard_examples)*100:.1f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("Sample Hard Examples:\n")
    
    # Show 3 examples from each category
    for error_type, examples in sorted(categories.items(), key=lambda x: -len(x[1])):
        report.append(f"\n{error_type.upper()} ({len(examples)} examples):")
        for i, example in enumerate(examples[:3]):
            report.append(f"\n  Example {i+1}:")
            report.append(f"    Question: {example['entry'].get('question', 'N/A')[:100]}...")
            report.append(f"    Prediction: {example['prediction'][:150]}...")
            report.append(f"    Gold: {example['gold'][:150]}...")
    
    report_text = '\n'.join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)

def main():
    parser = argparse.ArgumentParser(
        description="Mine hard examples from validation failures"
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        required=True,
        help="Path to validation dataset (JSON/JSONL)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions file"
    )
    parser.add_argument(
        "--gold_standard",
        type=str,
        required=True,
        help="Path to gold standard SQL file"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="Path to existing training data (to augment)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_hard_examples_augmented.json",
        help="Path to output augmented training data"
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=3,
        help="Oversample factor for hard examples"
    )
    parser.add_argument(
        "--max_additions",
        type=int,
        default=None,
        help="Maximum number of hard examples to add"
    )
    parser.add_argument(
        "--use_execution",
        action="store_true",
        help="Use execution-based accuracy check"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="eval/data/database",
        help="Directory containing databases"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="hard_examples_report.txt",
        help="Path to error report file"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading validation data...")
    validation_data = load_validation_data(args.validation_data)
    
    print("Loading predictions...")
    predictions = load_predictions(args.predictions)
    
    print("Loading gold standard...")
    gold_standard = load_gold_standard(args.gold_standard)
    
    # Identify hard examples
    print("Identifying hard examples...")
    hard_examples = identify_hard_examples(
        validation_data,
        predictions,
        gold_standard,
        use_execution=args.use_execution,
        db_dir=args.db_dir if args.use_execution else None,
    )
    
    print(f"Found {len(hard_examples)} hard examples")
    
    # Create error report
    create_error_report(hard_examples, args.report)
    
    # Add to training data if requested
    if args.train_data:
        add_to_training_data(
            hard_examples,
            args.train_data,
            args.output,
            oversample_factor=args.oversample,
            max_additions=args.max_additions,
        )
    
    print("\nHard example mining completed!")

if __name__ == "__main__":
    main()










