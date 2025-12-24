"""
Manual test script for validation fixes.
Tests edge cases without pytest dependency.
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from helpers.analysis import (
    run_ml_analysis,
    find_optimal_codes,
    calculate_metrics_summary,
)


def test_empty_dataframe():
    """Test empty DataFrame validation."""
    print("Test 1: Empty DataFrame...")
    df = pd.DataFrame()

    try:
        run_ml_analysis(df, text_column='text', n_codes=5)
        print("  ❌ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "DataFrame is empty" in str(e):
            print(f"  ✅ PASSED: {e}")
            return True
        else:
            print(f"  ❌ FAILED: Wrong error message: {e}")
            return False


def test_undersized_dataset():
    """Test under-sized dataset validation."""
    print("\nTest 2: Under-sized dataset...")
    df = pd.DataFrame({
        'text': ['response 1', 'response 2', 'response 3']
    })

    try:
        run_ml_analysis(df, text_column='text', n_codes=10)
        print("  ❌ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "Dataset too small" in str(e):
            print(f"  ✅ PASSED: {e}")
            return True
        else:
            print(f"  ❌ FAILED: Wrong error message: {e}")
            return False


def test_find_optimal_too_small():
    """Test find_optimal_codes with too-small dataset."""
    print("\nTest 3: Find optimal with 1 row...")
    df = pd.DataFrame({
        'text': ['single response']
    })

    try:
        find_optimal_codes(df, text_column='text', min_codes=3)
        print("  ❌ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "too small" in str(e).lower():
            print(f"  ✅ PASSED: {e}")
            return True
        else:
            print(f"  ❌ FAILED: Wrong error message: {e}")
            return False


def test_division_by_zero_guard():
    """Test that empty results don't cause division by zero."""
    print("\nTest 4: Division by zero guard...")

    class MockCoder:
        def __init__(self):
            self.codebook = {'CODE_01': {'count': 0}}

    coder = MockCoder()
    results_df = pd.DataFrame()

    try:
        metrics = calculate_metrics_summary(coder, results_df)
        if (metrics['total_responses'] == 0 and
            metrics['avg_codes_per_response'] == 0.0 and
            metrics['coverage_pct'] == 0.0):
            print(f"  ✅ PASSED: Returns zeroed metrics safely")
            print(f"     Metrics: {metrics}")
            return True
        else:
            print(f"  ❌ FAILED: Unexpected metrics: {metrics}")
            return False
    except Exception as e:
        print(f"  ❌ FAILED: Raised exception: {e}")
        return False


def test_valid_dataset():
    """Test that valid dataset processes successfully."""
    print("\nTest 5: Valid dataset (should succeed)...")
    df = pd.DataFrame({
        'text': [
            'This is a response about machine learning and artificial intelligence',
            'Another response discussing data science and analytics tools',
            'A third response about cloud computing infrastructure',
            'More content about software development and engineering',
            'Final response covering cybersecurity and privacy',
            'Extra response for web development and frameworks',
            'Additional content about database design',
            'More discussion on DevOps practices',
            'Yet another response about mobile development',
            'Last response here about project management'
        ]
    })

    try:
        coder, results_df, metrics = run_ml_analysis(
            df, text_column='text', n_codes=3
        )

        if (coder is not None and
            results_df is not None and
            len(results_df) == 10 and
            metrics['total_responses'] == 10):
            print(f"  ✅ PASSED: Analysis completed successfully")
            print(f"     Total responses: {metrics['total_responses']}")
            print(f"     Coverage: {metrics.get('coverage_pct', 0):.1f}%")
            return True
        else:
            print(f"  ❌ FAILED: Unexpected results")
            return False
    except Exception as e:
        print(f"  ❌ FAILED: Raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_optimal_valid():
    """Test find_optimal_codes with valid dataset."""
    print("\nTest 6: Find optimal codes (should succeed)...")
    df = pd.DataFrame({
        'text': [
            'Response about machine learning and AI technology',
            'Data science and analytics discussion here',
            'Cloud computing and infrastructure topics',
            'Software development best practices',
            'Cybersecurity and privacy concerns discussion',
            'DevOps and continuous integration methods',
            'Mobile app development strategies',
            'Web development frameworks overview',
            'Database design patterns explained',
            'Agile project management techniques'
        ]
    })

    try:
        optimal_n, results = find_optimal_codes(
            df, text_column='text', min_codes=2, max_codes=5
        )

        if (optimal_n >= 2 and optimal_n <= 5 and
            'silhouette_scores' in results and
            'optimal_n_codes' in results):
            print(f"  ✅ PASSED: Found optimal codes")
            print(f"     Optimal n: {optimal_n}")
            print(f"     Best score: {results.get('best_silhouette_score', 0):.4f}")
            return True
        else:
            print(f"  ❌ FAILED: Unexpected results")
            return False
    except Exception as e:
        print(f"  ❌ FAILED: Raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Running Validation Fix Tests")
    print("=" * 60)

    results = []
    results.append(test_empty_dataframe())
    results.append(test_undersized_dataset())
    results.append(test_find_optimal_too_small())
    results.append(test_division_by_zero_guard())
    results.append(test_valid_dataset())
    results.append(test_find_optimal_valid())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} test(s) failed")
        sys.exit(1)
