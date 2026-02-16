"""
LDA Topic Overlap Analysis
Runs LDA on all 6 datasets and examines co-occurrence heatmaps for overlapping themes.
"""

import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

warnings.filterwarnings('ignore')


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())


def run_lda_analysis(df, response_col='response', n_codes=10, min_confidence=0.15):
    """Run LDA and return codebook, assignments, co-occurrence data."""
    processed = [preprocess_text(r) for r in df[response_col]]

    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    feature_matrix = vectorizer.fit_transform(processed)

    lda = LatentDirichletAllocation(
        n_components=n_codes,
        random_state=42,
        max_iter=30
    )
    doc_topic_matrix = lda.fit_transform(feature_matrix)

    feature_names = vectorizer.get_feature_names_out()

    # Build codebook
    codebook = {}
    for idx in range(n_codes):
        topic_weights = lda.components_[idx]
        top_indices = topic_weights.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        label = ' '.join(top_words[:3]).title()
        codebook[f"TOPIC_{idx+1:02d}"] = {
            'label': label,
            'keywords': top_words,
            'top_weights': [float(topic_weights[i]) for i in top_indices[:5]]
        }

    # Assign codes (LDA gives soft assignments - use threshold)
    assignments = []
    for doc_idx, topic_dist in enumerate(doc_topic_matrix):
        doc_codes = []
        for topic_idx, weight in enumerate(topic_dist):
            if weight >= min_confidence:
                doc_codes.append(f"TOPIC_{topic_idx+1:02d}")
        assignments.append(doc_codes)

    # Co-occurrence matrix
    topic_ids = sorted(codebook.keys())
    n = len(topic_ids)
    tid_to_idx = {t: i for i, t in enumerate(topic_ids)}
    cooccur = np.zeros((n, n))

    for doc_codes in assignments:
        for c1, c2 in combinations(doc_codes, 2):
            i, j = tid_to_idx[c1], tid_to_idx[c2]
            cooccur[i, j] += 1
            cooccur[j, i] += 1
        for c in doc_codes:
            i = tid_to_idx[c]
            cooccur[i, i] += 1

    labels = [codebook[t]['label'] for t in topic_ids]
    cooccur_df = pd.DataFrame(cooccur, index=labels, columns=labels)

    # Co-occurrence pairs
    pairs = Counter()
    for doc_codes in assignments:
        for i_c, c1 in enumerate(doc_codes):
            for c2 in doc_codes[i_c+1:]:
                pair = tuple(sorted([c1, c2]))
                pairs[pair] += 1

    pair_data = []
    for (c1, c2), count in pairs.most_common():
        if count >= 2:
            pair_data.append({
                'Topic 1': c1,
                'Label 1': codebook[c1]['label'],
                'Topic 2': c2,
                'Label 2': codebook[c2]['label'],
                'Co-occurrence Count': count,
                'Percentage': (count / len(df)) * 100
            })
    pairs_df = pd.DataFrame(pair_data)

    # Stats
    total_multi = sum(1 for a in assignments if len(a) > 1)
    avg_topics = np.mean([len(a) for a in assignments])

    # Topic-term overlap analysis: find terms shared across topics
    shared_terms = {}
    for i in range(n_codes):
        wi = set(codebook[f"TOPIC_{i+1:02d}"]['keywords'][:7])
        for j in range(i+1, n_codes):
            wj = set(codebook[f"TOPIC_{j+1:02d}"]['keywords'][:7])
            overlap = wi & wj
            if overlap:
                shared_terms[(f"TOPIC_{i+1:02d}", f"TOPIC_{j+1:02d}")] = overlap

    return {
        'codebook': codebook,
        'assignments': assignments,
        'cooccur_df': cooccur_df,
        'pairs_df': pairs_df,
        'doc_topic_matrix': doc_topic_matrix,
        'total_responses': len(df),
        'multi_coded': total_multi,
        'avg_topics_per_response': avg_topics,
        'shared_terms': shared_terms
    }


# ============================================================
# Run analysis on all 6 datasets
# ============================================================

datasets = {
    'Healthcare Patient Feedback': ('data/Healthcare_Patient_Feedback_300.csv', 'response'),
    'Market Research Survey': ('data/Market_Research_Survey_300.csv', 'response'),
    'Psychology Wellbeing Study': ('data/Psychology_Wellbeing_Study_300.csv', 'response'),
    'Remote Work Experiences': ('data/Remote_Work_Experiences_200.csv', 'response'),
    'Cricket Responses': ('data/cricket_responses.csv', 'response'),
    'Fashion Responses': ('data/fashion_responses.csv', 'response'),
}

all_results = {}

for name, (path, col) in datasets.items():
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    print(f"  Responses: {len(df)}")

    result = run_lda_analysis(df, response_col=col, n_codes=8, min_confidence=0.15)
    all_results[name] = result

    print(f"  Multi-coded responses: {result['multi_coded']} ({result['multi_coded']/result['total_responses']*100:.1f}%)")
    print(f"  Avg topics per response: {result['avg_topics_per_response']:.2f}")

    # Print codebook
    print(f"\n  --- LDA Topics Discovered ---")
    for tid, info in result['codebook'].items():
        print(f"  {tid}: {info['label']}")
        print(f"         Keywords: {', '.join(info['keywords'][:7])}")

    # Print co-occurrence pairs (top 10)
    if len(result['pairs_df']) > 0:
        print(f"\n  --- Top Co-occurring Topic Pairs (from heatmap) ---")
        for _, row in result['pairs_df'].head(10).iterrows():
            print(f"  [{row['Co-occurrence Count']:3.0f} responses, {row['Percentage']:5.1f}%] "
                  f"{row['Label 1']}  <-->  {row['Label 2']}")
    else:
        print(f"\n  No co-occurring topic pairs found.")

    # Print shared vocabulary terms
    if result['shared_terms']:
        print(f"\n  --- Shared Vocabulary Across Topics (keyword overlap) ---")
        for (t1, t2), terms in sorted(result['shared_terms'].items(),
                                        key=lambda x: len(x[1]), reverse=True):
            l1 = result['codebook'][t1]['label']
            l2 = result['codebook'][t2]['label']
            print(f"  {l1} & {l2}: shared words = {terms}")

    # Print the co-occurrence matrix
    print(f"\n  --- Co-occurrence Matrix (heatmap values) ---")
    cooccur = result['cooccur_df']
    # Show off-diagonal values > 0
    n = len(cooccur)
    hot_cells = []
    for i in range(n):
        for j in range(i+1, n):
            val = cooccur.iloc[i, j]
            if val > 0:
                hot_cells.append((cooccur.index[i], cooccur.columns[j], val))

    hot_cells.sort(key=lambda x: x[2], reverse=True)
    if hot_cells:
        print(f"  {'Topic A':<35} {'Topic B':<35} {'Co-occurrences':>15}")
        print(f"  {'-'*35} {'-'*35} {'-'*15}")
        for a, b, v in hot_cells[:15]:
            print(f"  {a:<35} {b:<35} {v:>15.0f}")
    else:
        print(f"  No off-diagonal co-occurrences found (topics are fully distinct)")


# ============================================================
# CROSS-DATASET SUMMARY
# ============================================================
print(f"\n\n{'='*70}")
print(f"CROSS-DATASET SUMMARY: OVERLAPPING THEMES IN LDA CO-OCCURRENCE HEATMAPS")
print(f"{'='*70}")

for name, result in all_results.items():
    n_pairs = len(result['pairs_df'])
    n_shared = len(result['shared_terms'])
    pct_multi = result['multi_coded'] / result['total_responses'] * 100

    has_strong_overlap = False
    if n_pairs > 0:
        max_cooccur_pct = result['pairs_df']['Percentage'].max()
        has_strong_overlap = max_cooccur_pct > 10
    else:
        max_cooccur_pct = 0

    print(f"\n  {name}:")
    print(f"    Co-occurring pairs: {n_pairs}")
    print(f"    Max co-occurrence: {max_cooccur_pct:.1f}% of responses")
    print(f"    Multi-coded responses: {pct_multi:.1f}%")
    print(f"    Keyword overlap pairs: {n_shared}")
    print(f"    STRONG OVERLAP IN HEATMAP: {'YES' if has_strong_overlap else 'No'}")

    if has_strong_overlap:
        print(f"    Top overlapping themes:")
        for _, row in result['pairs_df'].head(3).iterrows():
            print(f"      - {row['Label 1']} + {row['Label 2']} ({row['Percentage']:.1f}%)")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
