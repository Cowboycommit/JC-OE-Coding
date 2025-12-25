"""
Content Quality Assessment Demonstration.

This example demonstrates how to use the ContentQualityFilter
integrated with DataLoader to assess and flag non-analytic responses.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import importlib.util

# Direct import to avoid numpy issues
spec_cq = importlib.util.spec_from_file_location('content_quality', '../src/content_quality.py')
content_quality = importlib.util.module_from_spec(spec_cq)
spec_cq.loader.exec_module(content_quality)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run content quality assessment demonstration."""
    print("=" * 80)
    print("Content Quality Assessment Demonstration")
    print("=" * 80)
    print()

    # Create filter
    filter = content_quality.ContentQualityFilter()

    # Sample survey responses (mix of good and problematic)
    sample_responses = [
        "I really enjoy working from home because it gives me flexibility.",
        "n/a",
        "Remote work has improved my work-life balance significantly.",
        "test",
        "idk",
        "The ability to work from anywhere has been life-changing for me.",
        "",
        "asdfgh",
        "Work from home is good because I can focus better and avoid commuting.",
        "123",
        "Not applicable to my situation",
        "ok",
        "I appreciate the quiet environment at home for concentrated work.",
        "qwerty response here",
        "very very very very very very same word",
        "Hola como estas bien gracias",
        "Remote work allows me to spend more time with my family.",
        "???",
        "The flexibility of remote work has helped me manage my time better.",
        "test test test",
    ]

    print(f"Assessing {len(sample_responses)} responses...\n")

    # Assess each response
    results = []
    for i, text in enumerate(sample_responses, 1):
        assessment = filter.assess_signal(text)
        results.append({
            'id': i,
            'text': text,
            'assessment': assessment
        })

        # Display result
        status = "✓ ANALYTIC" if assessment['is_analytic'] else "✗ FLAGGED"
        print(f"{i:2d}. [{status}] (conf: {assessment['confidence']:.2f})")
        print(f"    Text: \"{text[:60]}\"" + ("..." if len(text) > 60 else ""))
        if not assessment['is_analytic']:
            print(f"    Flags: {', '.join(assessment['flags'])}")
            print(f"    Reason: {assessment['reason']}")
            print(f"    Recommendation: {assessment['recommendation']}")
        print()

    # Summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    total = len(results)
    analytic = sum(1 for r in results if r['assessment']['is_analytic'])
    non_analytic = total - analytic

    print(f"\nTotal responses: {total}")
    print(f"Analytic responses: {analytic} ({analytic/total*100:.1f}%)")
    print(f"Non-analytic responses: {non_analytic} ({non_analytic/total*100:.1f}%)")

    # Recommendation breakdown
    recommendations = {}
    for r in results:
        rec = r['assessment']['recommendation']
        recommendations[rec] = recommendations.get(rec, 0) + 1

    print(f"\nRecommendations:")
    for rec, count in sorted(recommendations.items()):
        print(f"  {rec}: {count} ({count/total*100:.1f}%)")

    # Flag statistics
    all_flags = []
    for r in results:
        all_flags.extend(r['assessment']['flags'])

    flag_counts = {}
    for flag in all_flags:
        flag_counts[flag] = flag_counts.get(flag, 0) + 1

    print(f"\nFlag Distribution:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {flag}: {count}")

    # Performance test
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    import time

    # Generate 1000 test responses
    test_responses = [f"This is test response number {i} with some content" for i in range(1000)]

    start = time.time()
    filter.batch_assess(test_responses)
    elapsed = time.time() - start

    print(f"\nProcessed 1000 responses in {elapsed*1000:.2f}ms")
    print(f"Average: {elapsed*1000/1000:.3f}ms per response")
    print(f"Requirement: <100ms per 1000 responses")
    print(f"Status: {'✓ PASS' if elapsed < 0.1 else '✗ FAIL'}")

    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
