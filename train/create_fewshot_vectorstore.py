"""
Create Vector Store for Dynamic Few-shot Example Retrieval

This script indexes training examples in a vector store for semantic similarity
search. Use this for dynamic few-shot retrieval during inference.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON or JSONL"""
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
        raise ValueError(f"Unsupported format: {file_path}")

def extract_question(entry: Dict) -> str:
    """Extract question from entry"""
    if 'question' in entry:
        return entry['question']
    
    # Try to extract from text field
    text = entry.get('text', '')
    if '### Instruction:' in text:
        # Extract question from instruction
        import re
        match = re.search(r'Convert text to \w+:\s*(.+?)(?:\s+\||\n)', text)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_sql(entry: Dict) -> str:
    """Extract SQL from entry"""
    if 'ground_truth' in entry:
        sql = entry['ground_truth']
        # Remove skeleton part if present
        if '|' in sql:
            sql = sql.split('|')[-1].strip()
        return sql
    
    # Try to extract from text field
    text = entry.get('text', '')
    if '### Response:' in text:
        response = text.split('### Response:')[-1].strip()
        if '|' in response:
            response = response.split('|')[-1].strip()
        return response
    
    return ""

def extract_schema(entry: Dict) -> str:
    """Extract schema info from entry"""
    if 'db_info' in entry:
        return entry['db_info']
    
    # Try to extract from text field
    text = entry.get('text', '')
    if '### Instruction:' in text:
        # Schema is usually between "Convert text to" and question
        import re
        parts = text.split('### Instruction:')
        if len(parts) > 1:
            instruction = parts[1]
            # Extract schema part
            return instruction.split('Convert text to')[0] if 'Convert text to' in instruction else ""
    
    return ""

def create_vector_store(
    train_data: List[Dict],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "./vector_store",
):
    """
    Create vector store from training examples
    
    Args:
        train_data: List of training examples
        embedding_model: Name of embedding model
        output_dir: Directory to save vector store
    """
    print(f"Loading embedding model: {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    
    # Extract examples
    examples = []
    for entry in train_data:
        question = extract_question(entry)
        sql = extract_sql(entry)
        schema = extract_schema(entry)
        db_id = entry.get('db_id', '')
        
        if question and sql:
            examples.append({
                'question': question,
                'sql': sql,
                'schema': schema,
                'db_id': db_id,
                'full_entry': entry,  # Keep full entry for metadata
            })
    
    print(f"Found {len(examples)} valid examples")
    
    # Create embeddings
    print("Creating embeddings...")
    questions = [ex['question'] for ex in examples]
    embeddings = encoder.encode(questions, show_progress_bar=True)
    
    # Save vector store
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings and metadata
    np.save(output_path / "embeddings.npy", embeddings)
    
    metadata = {
        'examples': examples,
        'embedding_model': embedding_model,
        'num_examples': len(examples),
    }
    
    with open(output_path / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    # Also save encoder model info
    with open(output_path / "config.json", 'w') as f:
        json.dump({
            'embedding_model': embedding_model,
            'num_examples': len(examples),
            'embedding_dim': embeddings.shape[1],
        }, f, indent=2)
    
    print(f"Vector store saved to: {output_dir}")
    print(f"  - Embeddings: {embeddings.shape}")
    print(f"  - Examples: {len(examples)}")
    
    return encoder, embeddings, examples

def search_similar_examples(
    query: str,
    encoder: SentenceTransformer,
    embeddings: np.ndarray,
    examples: List[Dict],
    top_k: int = 3,
    db_id_filter: str = None,
):
    """
    Search for similar examples
    
    Args:
        query: Question to search for
        encoder: SentenceTransformer model
        embeddings: Pre-computed embeddings
        examples: List of example dictionaries
        top_k: Number of similar examples to return
        db_id_filter: Optional database ID to filter by
    
    Returns:
        List of similar examples (question, sql, schema, similarity_score)
    """
    # Encode query
    query_embedding = encoder.encode([query])
    
    # Compute similarities
    similarities = np.dot(query_embedding, embeddings.T)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Filter by db_id if specified
    if db_id_filter:
        filtered_indices = [
            idx for idx in top_indices
            if examples[idx]['db_id'] == db_id_filter
        ]
        if filtered_indices:
            top_indices = filtered_indices[:top_k]
    
    # Return similar examples
    results = []
    for idx in top_indices:
        ex = examples[idx]
        results.append({
            'question': ex['question'],
            'sql': ex['sql'],
            'schema': ex.get('schema', ''),
            'db_id': ex['db_id'],
            'similarity': float(similarities[idx]),
            'rank': len(results) + 1,
        })
    
    return results

def format_examples_for_prompt(examples: List[Dict], max_examples: int = 3) -> str:
    """Format examples for use in prompt"""
    formatted = []
    for ex in examples[:max_examples]:
        formatted.append(f"""
Example Question: {ex['question']}
Example SQL: {ex['sql']}
""")
    return "\n".join(formatted)

def main():
    parser = argparse.ArgumentParser(
        description="Create vector store for few-shot example retrieval"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON/JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vector_store",
        help="Output directory for vector store"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--test_query",
        type=str,
        default=None,
        help="Test query to search for similar examples (optional)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of similar examples to retrieve (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from: {args.train_data}")
    train_data = load_dataset(args.train_data)
    print(f"Loaded {len(train_data)} examples")
    
    # Create vector store
    encoder, embeddings, examples = create_vector_store(
        train_data,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
    )
    
    # Test search if query provided
    if args.test_query:
        print(f"\nTesting search with query: {args.test_query}")
        similar = search_similar_examples(
            args.test_query,
            encoder,
            embeddings,
            examples,
            top_k=args.top_k,
        )
        
        print(f"\nTop {args.top_k} similar examples:")
        for ex in similar:
            print(f"\n  Rank {ex['rank']} (similarity: {ex['similarity']:.3f}):")
            print(f"    Question: {ex['question']}")
            print(f"    SQL: {ex['sql'][:100]}...")
            print(f"    DB: {ex['db_id']}")
    
    print("\n✅ Vector store creation complete!")
    print(f"\nTo use in inference, load with:")
    print(f"  from train.create_fewshot_vectorstore import load_vector_store")
    print(f"  encoder, embeddings, examples = load_vector_store('{args.output_dir}')")

def load_vector_store(store_dir: str):
    """Load saved vector store"""
    from sentence_transformers import SentenceTransformer
    
    store_path = Path(store_dir)
    
    # Load config
    with open(store_path / "config.json") as f:
        config = json.load(f)
    
    # Load encoder
    encoder = SentenceTransformer(config['embedding_model'])
    
    # Load embeddings
    embeddings = np.load(store_path / "embeddings.npy")
    
    # Load metadata
    with open(store_path / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
        examples = metadata['examples']
    
    return encoder, embeddings, examples

if __name__ == "__main__":
    main()










