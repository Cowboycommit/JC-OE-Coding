"""
Test script to verify co-occurrence heatmap optimization produces correct results.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def old_cooccurrence(results_df, codebook):
    """Original O(n_codes^2 * n_responses) implementation."""
    codes = list(codebook.keys())
    n = len(codes)
    cooccur = np.zeros((n, n))

    for assigned_codes in results_df['assigned_codes']:
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                if code1 in assigned_codes and code2 in assigned_codes:
                    cooccur[i, j] += 1

    return cooccur


def new_cooccurrence(results_df, codebook):
    """Optimized O(n_responses * avg_codes^2) implementation."""
    codes = list(codebook.keys())
    n = len(codes)
    code_to_idx = {code: i for i, code in enumerate(codes)}
    cooccur = np.zeros((n, n))

    for assigned_codes in results_df['assigned_codes']:
        # Only process pairs of codes that were actually assigned
        for code1, code2 in combinations(assigned_codes, 2):
            i, j = code_to_idx[code1], code_to_idx[code2]
            cooccur[i, j] += 1
            cooccur[j, i] += 1  # Symmetric matrix

        # Diagonal: each code co-occurs with itself
        for code in assigned_codes:
            i = code_to_idx[code]
            cooccur[i, i] += 1

    return cooccur


def test_cooccurrence():
    """Test that both implementations produce identical results."""
    # Create test data
    codebook = {
        'CODE_01': {'label': 'Code 1'},
        'CODE_02': {'label': 'Code 2'},
        'CODE_03': {'label': 'Code 3'},
        'CODE_04': {'label': 'Code 4'},
        'CODE_05': {'label': 'Code 5'},
    }

    # Test cases with varying code assignments
    test_cases = [
        # Single code per response
        [['CODE_01'], ['CODE_02'], ['CODE_03']],
        # Multiple codes per response
        [['CODE_01', 'CODE_02'], ['CODE_02', 'CODE_03'], ['CODE_01', 'CODE_03']],
        # All codes in one response
        [['CODE_01', 'CODE_02', 'CODE_03', 'CODE_04', 'CODE_05']],
        # Mixed scenarios
        [['CODE_01'], ['CODE_01', 'CODE_02'], ['CODE_02', 'CODE_03', 'CODE_04'], ['CODE_05']],
        # Empty codes
        [[], ['CODE_01'], []],
    ]

    for i, assigned_codes in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {assigned_codes}")
        results_df = pd.DataFrame({'assigned_codes': assigned_codes})

        old_result = old_cooccurrence(results_df, codebook)
        new_result = new_cooccurrence(results_df, codebook)

        if np.allclose(old_result, new_result):
            print("✓ PASS: Results match")
        else:
            print("✗ FAIL: Results differ!")
            print(f"Old:\n{old_result}")
            print(f"New:\n{new_result}")
            print(f"Diff:\n{old_result - new_result}")
            return False

    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    # Large dataset test
    import time
    n_responses = 1000
    n_codes = 50
    avg_codes_per_response = 3

    # Generate large test dataset
    large_codebook = {f'CODE_{i:02d}': {'label': f'Code {i}'} for i in range(1, n_codes + 1)}
    code_list = list(large_codebook.keys())

    np.random.seed(42)
    large_assigned = []
    for _ in range(n_responses):
        n_assigned = np.random.poisson(avg_codes_per_response)
        n_assigned = max(0, min(n_assigned, n_codes))
        if n_assigned > 0:
            codes = list(np.random.choice(code_list, n_assigned, replace=False))
            large_assigned.append(codes)
        else:
            large_assigned.append([])

    large_df = pd.DataFrame({'assigned_codes': large_assigned})

    print(f"\nDataset: {n_responses} responses, {n_codes} codes")
    print(f"Average codes per response: {np.mean([len(c) for c in large_assigned]):.2f}")

    # Time old implementation
    start = time.time()
    old_result = old_cooccurrence(large_df, large_codebook)
    old_time = time.time() - start
    print(f"\nOld implementation: {old_time:.4f}s")
    print(f"  Complexity: O({n_codes}² × {n_responses}) = {n_codes**2 * n_responses:,} iterations")

    # Time new implementation
    start = time.time()
    new_result = new_cooccurrence(large_df, large_codebook)
    new_time = time.time() - start
    print(f"New implementation: {new_time:.4f}s")
    avg_ops = sum(len(c)**2 for c in large_assigned)
    print(f"  Complexity: O({n_responses} × avg_codes²) ≈ {avg_ops:,} operations")

    # Verify results match
    if np.allclose(old_result, new_result):
        print(f"\n✓ Results match!")
        speedup = old_time / new_time
        print(f"✓ Speedup: {speedup:.2f}x faster")
        print(f"✓ Time saved: {(old_time - new_time)*1000:.2f}ms")
    else:
        print("\n✗ Results differ on large dataset!")
        return False

    return True


if __name__ == '__main__':
    print("="*60)
    print("CO-OCCURRENCE HEATMAP OPTIMIZATION TEST")
    print("="*60)

    if test_cooccurrence():
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        exit(0)
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        exit(1)
