#!/usr/bin/env python3
"""
Generate example METHODS.md file demonstrating the MethodsDocGenerator.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.methods_documentation import MethodsDocGenerator, export_methods_to_file


class MockMLOpenCoder:
    """Mock coder for example generation."""
    def __init__(self, n_codes=10, method='tfidf_kmeans', min_confidence=0.3):
        self.n_codes = n_codes
        self.method = method
        self.min_confidence = min_confidence
        self.codebook = {
            'CODE_01': {
                'label': 'Remote Work Flexibility',
                'keywords': ['remote', 'work', 'home', 'flexible', 'office'],
                'count': 45,
                'examples': [],
                'avg_confidence': 0.78
            },
            'CODE_02': {
                'label': 'Career Growth Opportunities',
                'keywords': ['career', 'growth', 'development', 'promotion', 'learning'],
                'count': 38,
                'examples': [],
                'avg_confidence': 0.72
            },
            'CODE_03': {
                'label': 'Work Life Balance',
                'keywords': ['balance', 'life', 'family', 'time', 'personal'],
                'count': 42,
                'examples': [],
                'avg_confidence': 0.69
            },
            'CODE_04': {
                'label': 'Compensation Benefits Satisfaction',
                'keywords': ['salary', 'pay', 'benefits', 'compensation', 'insurance'],
                'count': 31,
                'examples': [],
                'avg_confidence': 0.81
            },
            'CODE_05': {
                'label': 'Team Collaboration Culture',
                'keywords': ['team', 'collaboration', 'culture', 'colleagues', 'support'],
                'count': 29,
                'examples': [],
                'avg_confidence': 0.65
            },
            'CODE_06': {
                'label': 'Management Leadership Quality',
                'keywords': ['management', 'leadership', 'manager', 'supervisor', 'support'],
                'count': 27,
                'examples': [],
                'avg_confidence': 0.74
            },
            'CODE_07': {
                'label': 'Job Security Stability',
                'keywords': ['security', 'stability', 'layoffs', 'contract', 'permanent'],
                'count': 19,
                'examples': [],
                'avg_confidence': 0.71
            },
            'CODE_08': {
                'label': 'Meaningful Impact Work',
                'keywords': ['meaningful', 'impact', 'purpose', 'mission', 'values'],
                'count': 23,
                'examples': [],
                'avg_confidence': 0.67
            },
            'CODE_09': {
                'label': 'Commute Location Convenience',
                'keywords': ['commute', 'location', 'travel', 'distance', 'parking'],
                'count': 12,
                'examples': [],
                'avg_confidence': 0.82
            },
            'CODE_10': {
                'label': 'Technology Tools Resources',
                'keywords': ['technology', 'tools', 'equipment', 'software', 'resources'],
                'count': 15,
                'examples': [],
                'avg_confidence': 0.58
            }
        }


def main():
    """Generate example METHODS.md file."""
    print("Generating example METHODS.md file...")

    # Create mock data
    n_responses = 300
    results_df = pd.DataFrame({
        'response_id': range(n_responses),
        'response_text': [f'Sample response {i}' for i in range(n_responses)],
        'assigned_codes': [
            ['CODE_01', 'CODE_02'] if i % 3 == 0 else
            ['CODE_03'] if i % 3 == 1 else
            [] if i % 20 == 0 else
            ['CODE_04', 'CODE_05', 'CODE_06'] if i % 10 == 0 else
            ['CODE_0' + str((i % 10) + 1)]
            for i in range(n_responses)
        ],
        'confidence_scores': [
            [0.75, 0.68] if i % 3 == 0 else
            [0.71] if i % 3 == 1 else
            [] if i % 20 == 0 else
            [0.72, 0.65, 0.58] if i % 10 == 0 else
            [0.60 + (i % 10) * 0.02]
            for i in range(n_responses)
        ],
        'num_codes': [
            2 if i % 3 == 0 else
            1 if i % 3 == 1 else
            0 if i % 20 == 0 else
            3 if i % 10 == 0 else
            1
            for i in range(n_responses)
        ]
    })

    coder = MockMLOpenCoder(n_codes=10, method='tfidf_kmeans', min_confidence=0.3)

    # Calculate metrics
    all_confidences = [conf for confs in results_df['confidence_scores'] for conf in confs]
    metrics = {
        'total_responses': n_responses,
        'method': 'tfidf_kmeans',
        'n_codes': 10,
        'avg_codes_per_response': results_df['num_codes'].mean(),
        'coverage_pct': (results_df['num_codes'] > 0).sum() / n_responses * 100,
        'uncoded_count': (results_df['num_codes'] == 0).sum(),
        'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
        'min_confidence': np.min(all_confidences) if all_confidences else 0,
        'max_confidence': np.max(all_confidences) if all_confidences else 0,
        'silhouette_score': 0.42
    }

    # Preprocessing params
    preprocessing_params = {
        'remove_nulls': True,
        'min_length': 5,
        'remove_duplicates': False,
        'stop_words': 'english'
    }

    # Generate methods documentation
    generator = MethodsDocGenerator(project_name="Employee Satisfaction Survey Analysis")
    methods = generator.generate_methods_section(
        coder,
        results_df,
        metrics,
        preprocessing_params
    )

    # Audit for objectivity claims
    print("\nRunning objectivity claims audit...")
    passed, violations = generator.audit_objectivity_claims(methods)
    if passed:
        print("✓ Objectivity claims audit PASSED (no violations found)")
    else:
        print(f"✗ Objectivity claims audit FAILED ({len(violations)} violations)")
        for v in violations:
            print(f"  - {v['phrase']}: {v['context'][:80]}...")

    # Generate BibTeX citations
    print("\nGenerating BibTeX citations...")
    bibtex = generator.generate_bibtex_citations('tfidf_kmeans')

    # Generate parameter log
    print("\nGenerating parameter log...")
    param_log = generator.generate_parameter_log(coder, metrics, preprocessing_params)

    # Export to file
    output_path = "METHODS.md"
    export_methods_to_file(methods, output_path)
    print(f"\n✓ Methods documentation exported to: {output_path}")

    # Also export BibTeX
    bibtex_path = "CITATIONS.bib"
    with open(bibtex_path, 'w', encoding='utf-8') as f:
        f.write(bibtex)
    print(f"✓ BibTeX citations exported to: {bibtex_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Methods documentation length: {len(methods):,} characters")
    print(f"Number of sections: {methods.count('## ')}")
    print(f"Number of assumptions documented: {methods.count('Assumption')}")
    print(f"Objectivity audit: {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"Human review mentions: {methods.lower().count('human review')}")
    print(f"Limitations mentioned: {methods.lower().count('cannot')}")
    print("=" * 60)

    print("\nExample generation complete!")
    print(f"\nView the generated file: cat {output_path}")
    print(f"View citations: cat {bibtex_path}")


if __name__ == '__main__':
    main()
