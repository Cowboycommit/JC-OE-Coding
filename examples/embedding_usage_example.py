"""
Example: Using Different Embedding Methods for ML Open-Ended Coding

This script demonstrates how to use the new semantic embedding methods
as alternatives to the default TF-IDF representation.

All embedding methods work offline (no API keys required).
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from helpers.analysis import run_ml_analysis

# Sample survey responses about remote work
sample_responses = [
    "Remote work gives me flexibility to manage my schedule",
    "Working from home improves my work-life balance",
    "I love the autonomy of remote work",
    "Office collaboration is essential for creativity",
    "In-person meetings build better team relationships",
    "Remote work can feel isolating sometimes",
    "Hybrid model combines the best of both worlds",
    "I prefer being in the office with my colleagues",
    "Video calls are exhausting compared to face-to-face",
    "Remote work saves me 2 hours of commute daily",
    "Flexibility to work anywhere is amazing",
    "Home office setup is more comfortable than the office",
    "I miss spontaneous conversations at the office",
    "Working remotely lets me focus without interruptions",
    "Team bonding happens naturally in the office",
    "Remote work allows me to live anywhere I want",
    "Office environment provides structure and routine",
    "I can balance family and work better remotely",
    "Collaboration tools make remote teamwork easy",
    "Physical presence matters for complex discussions"
]

# Create DataFrame
df = pd.DataFrame({'response': sample_responses})

print("=" * 70)
print("EMBEDDING METHODS COMPARISON")
print("=" * 70)
print()

# ============================================================================
# Example 1: Default TF-IDF (Backward Compatible)
# ============================================================================

print("1. TF-IDF (Default - Backward Compatible)")
print("-" * 70)
print("Method: Keyword-based, bag-of-words")
print("Best for: Fast exploration, interpretable results")
print()

coder1, results1, metrics1 = run_ml_analysis(
    df,
    text_column='response',
    n_codes=3,
    method='tfidf_kmeans',
    # representation='tfidf'  # This is the default, can be omitted
)

print(f"✓ Completed in {metrics1['execution_time']:.2f} seconds")
print(f"  - Representation: {metrics1['representation']}")
print(f"  - Coverage: {metrics1.get('coverage_pct', 0):.1f}%")
if 'silhouette_score' in metrics1:
    print(f"  - Silhouette Score: {metrics1['silhouette_score']:.3f}")
print()

# Show top codes
print("Top Codes Discovered:")
for code_id, info in list(coder1.codebook.items())[:3]:
    print(f"  - {code_id}: '{info['label']}' (count={info['count']})")
print()
print()

# ============================================================================
# Example 2: Word2Vec Embeddings
# ============================================================================

print("2. Word2Vec Embeddings (Semantic Similarity)")
print("-" * 70)
print("Method: Average word vectors, captures synonyms")
print("Best for: Medium datasets (1000+), semantic themes")
print()

coder2, results2, metrics2 = run_ml_analysis(
    df,
    text_column='response',
    n_codes=3,
    method='tfidf_kmeans',
    representation='word2vec',
    embedding_kwargs={
        'vector_size': 100,  # Embedding dimension
        'window': 5,         # Context window size
        'min_count': 1,      # Include rare words
        'epochs': 10         # Training iterations
    }
)

print(f"✓ Completed in {metrics2['execution_time']:.2f} seconds")
print(f"  - Representation: {metrics2['representation']}")
print(f"  - Coverage: {metrics2.get('coverage_pct', 0):.1f}%")
if 'silhouette_score' in metrics2:
    print(f"  - Silhouette Score: {metrics2['silhouette_score']:.3f}")
print()

# Show top codes
print("Top Codes Discovered:")
for code_id, info in list(coder2.codebook.items())[:3]:
    print(f"  - {code_id}: '{info['label']}' (count={info['count']})")
print()
print()

# ============================================================================
# Example 3: FastText Embeddings
# ============================================================================

print("3. FastText Embeddings (Handles Typos)")
print("-" * 70)
print("Method: Subword-aware vectors, robust to misspellings")
print("Best for: Messy text, typos, compound words")
print()

coder3, results3, metrics3 = run_ml_analysis(
    df,
    text_column='response',
    n_codes=3,
    method='tfidf_kmeans',
    representation='fasttext',
    embedding_kwargs={
        'vector_size': 100,  # Embedding dimension
        'min_count': 1,      # Include rare words
        'min_n': 3,          # Min character n-gram
        'max_n': 6           # Max character n-gram
    }
)

print(f"✓ Completed in {metrics3['execution_time']:.2f} seconds")
print(f"  - Representation: {metrics3['representation']}")
print(f"  - Coverage: {metrics3.get('coverage_pct', 0):.1f}%")
if 'silhouette_score' in metrics3:
    print(f"  - Silhouette Score: {metrics3['silhouette_score']:.3f}")
print()

# Show top codes
print("Top Codes Discovered:")
for code_id, info in list(coder3.codebook.items())[:3]:
    print(f"  - {code_id}: '{info['label']}' (count={info['count']})")
print()
print()

# ============================================================================
# Example 4: SentenceBERT (Requires Installation)
# ============================================================================

print("4. SentenceBERT Embeddings (Highest Quality)")
print("-" * 70)
print("Method: Pre-trained transformers, best semantic understanding")
print("Best for: Small datasets (<5000), maximum quality")
print("Requires: pip install sentence-transformers torch")
print()

try:
    import sentence_transformers

    coder4, results4, metrics4 = run_ml_analysis(
        df,
        text_column='response',
        n_codes=3,
        method='tfidf_kmeans',
        representation='sbert',
        embedding_kwargs={
            'model_name': 'all-MiniLM-L6-v2',  # Fast, good quality
            'device': 'cpu',                   # Use 'cuda' for GPU
            'batch_size': 16
        }
    )

    print(f"✓ Completed in {metrics4['execution_time']:.2f} seconds")
    print(f"  - Representation: {metrics4['representation']}")
    print(f"  - Coverage: {metrics4.get('coverage_pct', 0):.1f}%")
    if 'silhouette_score' in metrics4:
        print(f"  - Silhouette Score: {metrics4['silhouette_score']:.3f}")
    print()

    # Show top codes
    print("Top Codes Discovered:")
    for code_id, info in list(coder4.codebook.items())[:3]:
        print(f"  - {code_id}: '{info['label']}' (count={info['count']})")

except ImportError:
    print("⚠️  SentenceBERT not installed. Skipping...")
    print("   To use SBERT: pip install sentence-transformers torch")

print()
print()

# ============================================================================
# Performance Comparison
# ============================================================================

print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print()

print(f"{'Method':<20} {'Time (s)':<12} {'Silhouette':<12} {'Coverage %':<12}")
print("-" * 70)

def print_row(name, metrics):
    time_str = f"{metrics['execution_time']:.2f}"
    sil_str = f"{metrics.get('silhouette_score', 0):.3f}" if 'silhouette_score' in metrics else "N/A"
    cov_str = f"{metrics.get('coverage_pct', 0):.1f}"
    print(f"{name:<20} {time_str:<12} {sil_str:<12} {cov_str:<12}")

print_row("TF-IDF", metrics1)
print_row("Word2Vec", metrics2)
print_row("FastText", metrics3)

print()
print()

# ============================================================================
# Recommendations
# ============================================================================

print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()

print("📊 When to use each method:")
print()
print("  TF-IDF:")
print("    ✓ Large datasets (>5,000 responses)")
print("    ✓ Fast exploration needed")
print("    ✓ Interpretability is critical")
print("    ✓ Keyword-based analysis")
print()
print("  Word2Vec:")
print("    ✓ Medium datasets (1,000-5,000 responses)")
print("    ✓ Semantic similarity matters")
print("    ✓ Synonyms are important")
print("    ✓ Good balance of speed and quality")
print()
print("  FastText:")
print("    ✓ Messy text with typos")
print("    ✓ Rare or compound words")
print("    ✓ Informal language")
print("    ✓ Similar to Word2Vec but more robust")
print()
print("  SentenceBERT:")
print("    ✓ Small datasets (<5,000 responses)")
print("    ✓ Maximum semantic quality needed")
print("    ✓ Paraphrase detection")
print("    ✓ GPU available for faster processing")
print()

print("=" * 70)
print("For more details, see: documentation/EMBEDDING_METHODS.md")
print("=" * 70)
