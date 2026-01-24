"""
Snapshot Regression Tests for Text Preprocessing Pipeline.

This module provides snapshot-based regression testing for the preprocessing pipeline.
It compares current preprocessing outputs against golden baseline outputs to detect
unexpected changes in behavior.

Usage:
    # Run regression tests
    pytest tests/test_preprocessing_regression.py -v

    # Generate/update golden outputs (run this once to create baseline)
    pytest tests/test_preprocessing_regression.py::test_generate_golden_outputs -v --generate-golden

    # Or use the command-line function
    python -c "from tests.test_preprocessing_regression import generate_golden_outputs; generate_golden_outputs()"
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.text_preprocessor import TextPreprocessor, TextPreprocessingError


# =============================================================================
# CONSTANTS AND PATHS
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_TEXTS_PATH = FIXTURES_DIR / "sample_texts.json"
GOLDEN_OUTPUTS_PATH = FIXTURES_DIR / "golden_outputs.json"


# =============================================================================
# FIXTURE LOADING UTILITIES
# =============================================================================

def load_sample_texts() -> Dict[str, Any]:
    """Load sample texts from fixture file."""
    if not SAMPLE_TEXTS_PATH.exists():
        raise FileNotFoundError(
            f"Sample texts fixture not found at {SAMPLE_TEXTS_PATH}. "
            "Please create the fixture file first."
        )
    with open(SAMPLE_TEXTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_outputs() -> Dict[str, Any]:
    """Load golden outputs from fixture file."""
    if not GOLDEN_OUTPUTS_PATH.exists():
        raise FileNotFoundError(
            f"Golden outputs fixture not found at {GOLDEN_OUTPUTS_PATH}. "
            "Please run generate_golden_outputs() first."
        )
    with open(GOLDEN_OUTPUTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_golden_outputs(data: Dict[str, Any]) -> None:
    """Save golden outputs to fixture file."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_OUTPUTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Golden outputs saved to {GOLDEN_OUTPUTS_PATH}")


# =============================================================================
# PREPROCESSOR CONFIGURATIONS
# =============================================================================

def create_legacy_preprocessor() -> TextPreprocessor:
    """Create preprocessor with legacy mode settings (full preprocessing)."""
    return TextPreprocessor(
        use_gold_standard=True,
        expand_contractions=True,
        normalize_elongations=True,
        normalize_punctuation=True,
        standardize_urls=True,
        standardize_mentions=True,
        process_hashtags=True,
    )


def create_enhanced_preprocessor() -> TextPreprocessor:
    """Create preprocessor with enhanced mode settings (lighter preprocessing)."""
    return TextPreprocessor(
        use_gold_standard=True,
        expand_contractions=True,
        normalize_elongations=True,
        normalize_punctuation=True,
        standardize_urls=True,
        standardize_mentions=True,
        process_hashtags=True,
        expand_slang=False,
        detect_spam=False,
    )


# =============================================================================
# GOLDEN OUTPUT GENERATION
# =============================================================================

def generate_golden_outputs(force: bool = False) -> Dict[str, Any]:
    """
    Generate golden outputs from current preprocessing implementation.

    This function should be run once to create the baseline, and then
    again only when intentional changes are made to the preprocessing.

    Args:
        force: If True, overwrite existing golden outputs without confirmation.

    Returns:
        Dictionary containing the generated golden outputs.
    """
    # Check if golden outputs already exist with data
    if GOLDEN_OUTPUTS_PATH.exists() and not force:
        existing = load_golden_outputs()
        if existing.get("outputs", {}).get("legacy_mode"):
            print("WARNING: Golden outputs already exist and contain data.")
            print("Use force=True to overwrite, or delete the file manually.")
            response = input("Overwrite existing golden outputs? [y/N]: ")
            if response.lower() != "y":
                print("Aborted. Existing golden outputs preserved.")
                return existing

    # Load sample texts
    sample_data = load_sample_texts()
    test_cases = sample_data["test_cases"]

    # Create preprocessors
    legacy_preprocessor = create_legacy_preprocessor()
    enhanced_preprocessor = create_enhanced_preprocessor()

    # Generate outputs for each mode
    legacy_outputs = {}
    enhanced_outputs = {}

    for test_case in test_cases:
        test_id = test_case["id"]
        input_text = test_case["input"]

        # Legacy mode: full preprocessing with stopwords removal and lemmatization
        try:
            legacy_result = legacy_preprocessor.preprocess(
                input_text,
                remove_stopwords=True,
                lemmatize=True,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )
        except TextPreprocessingError:
            legacy_result = None

        legacy_outputs[test_id] = {
            "input": input_text,
            "output": legacy_result,
            "description": test_case.get("description", ""),
            "category": test_case.get("category", ""),
        }

        # Enhanced mode: normalization without stopwords removal or lemmatization
        try:
            enhanced_result = enhanced_preprocessor.preprocess(
                input_text,
                remove_stopwords=False,
                lemmatize=False,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )
        except TextPreprocessingError:
            enhanced_result = None

        enhanced_outputs[test_id] = {
            "input": input_text,
            "output": enhanced_result,
            "description": test_case.get("description", ""),
            "category": test_case.get("category", ""),
        }

    # Create golden outputs structure
    golden_data = {
        "description": "Golden outputs for preprocessing regression testing",
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "preprocessor_config": {
            "legacy_mode": {
                "use_gold_standard": True,
                "remove_stopwords": True,
                "lemmatize": True,
                "lowercase": True,
                "min_token_length": 2,
            },
            "enhanced_mode": {
                "use_gold_standard": True,
                "remove_stopwords": False,
                "lemmatize": False,
                "lowercase": True,
                "min_token_length": 2,
            },
        },
        "outputs": {
            "legacy_mode": legacy_outputs,
            "enhanced_mode": enhanced_outputs,
        },
    }

    # Save golden outputs
    save_golden_outputs(golden_data)

    print(f"Generated golden outputs for {len(test_cases)} test cases.")
    print(f"  - Legacy mode: {len(legacy_outputs)} outputs")
    print(f"  - Enhanced mode: {len(enhanced_outputs)} outputs")

    return golden_data


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def format_diff(test_id: str, expected: Optional[str], actual: Optional[str]) -> str:
    """Format a clear diff message for test failures."""
    lines = [
        f"\n{'='*60}",
        f"SNAPSHOT MISMATCH for test case: {test_id}",
        f"{'='*60}",
        "",
        "EXPECTED:",
        f"  {repr(expected)}",
        "",
        "ACTUAL:",
        f"  {repr(actual)}",
        "",
    ]

    if expected is not None and actual is not None:
        # Show character-by-character diff for non-None values
        if expected != actual:
            lines.append("DIFF ANALYSIS:")
            lines.append(f"  Expected length: {len(expected)}")
            lines.append(f"  Actual length:   {len(actual)}")

            # Find first difference
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    lines.append(f"  First diff at position {i}: {repr(e)} vs {repr(a)}")
                    lines.append(f"  Context: ...{expected[max(0,i-10):i+10]}...")
                    break
            else:
                if len(expected) != len(actual):
                    lines.append(f"  Strings differ only in length")

    lines.append("="*60)
    return "\n".join(lines)


def compare_outputs(
    test_id: str,
    expected: Optional[str],
    actual: Optional[str],
) -> Tuple[bool, str]:
    """
    Compare expected and actual outputs.

    Returns:
        Tuple of (match: bool, message: str)
    """
    if expected == actual:
        return True, f"Test {test_id}: PASS"

    return False, format_diff(test_id, expected, actual)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_texts() -> Dict[str, Any]:
    """Fixture to load sample texts."""
    return load_sample_texts()


@pytest.fixture
def golden_outputs() -> Dict[str, Any]:
    """Fixture to load golden outputs."""
    return load_golden_outputs()


@pytest.fixture
def legacy_preprocessor() -> TextPreprocessor:
    """Fixture for legacy mode preprocessor."""
    return create_legacy_preprocessor()


@pytest.fixture
def enhanced_preprocessor() -> TextPreprocessor:
    """Fixture for enhanced mode preprocessor."""
    return create_enhanced_preprocessor()


# =============================================================================
# GOLDEN OUTPUT GENERATION TEST
# =============================================================================

@pytest.mark.generate_golden
def test_generate_golden_outputs(request):
    """
    Generate golden outputs from current preprocessing.

    Run with: pytest tests/test_preprocessing_regression.py::test_generate_golden_outputs -v --generate-golden

    Or use the environment variable:
        GENERATE_GOLDEN=1 pytest tests/test_preprocessing_regression.py::test_generate_golden_outputs -v
    """
    import os

    # Check for --generate-golden flag (may not be registered in all pytest contexts)
    try:
        generate_flag = request.config.getoption("--generate-golden")
    except ValueError:
        generate_flag = False

    # Also check environment variable as alternative
    env_flag = os.environ.get("GENERATE_GOLDEN", "").lower() in ("1", "true", "yes")

    if not generate_flag and not env_flag:
        pytest.skip(
            "Run with --generate-golden flag or GENERATE_GOLDEN=1 env var to generate golden outputs"
        )

    golden_data = generate_golden_outputs(force=True)
    assert golden_data is not None
    assert "outputs" in golden_data
    assert "legacy_mode" in golden_data["outputs"]
    assert "enhanced_mode" in golden_data["outputs"]


# =============================================================================
# REGRESSION TESTS - LEGACY MODE
# =============================================================================

class TestLegacyModeRegression:
    """Regression tests for legacy preprocessing mode."""

    def test_golden_outputs_exist(self, golden_outputs):
        """Verify golden outputs file exists and has expected structure."""
        assert "outputs" in golden_outputs, "Golden outputs missing 'outputs' key"
        assert "legacy_mode" in golden_outputs["outputs"], "Golden outputs missing 'legacy_mode'"
        assert len(golden_outputs["outputs"]["legacy_mode"]) > 0, (
            "No legacy mode outputs found. Run generate_golden_outputs() first."
        )

    def test_sample_texts_match_golden(self, sample_texts, golden_outputs):
        """Verify all sample texts have corresponding golden outputs."""
        sample_ids = {tc["id"] for tc in sample_texts["test_cases"]}
        golden_ids = set(golden_outputs["outputs"]["legacy_mode"].keys())

        missing = sample_ids - golden_ids
        assert not missing, (
            f"Sample texts missing from golden outputs: {missing}. "
            "Run generate_golden_outputs() to update."
        )

    @pytest.mark.parametrize("category", [
        "simple",
        "contractions",
        "negations",
        "urls_emails",
        "numbers",
        "domain_specific",
        "mixed_case",
        "special_characters",
        "empty_whitespace",
        "long_text",
        "social_media",
    ])
    def test_legacy_mode_by_category(
        self,
        category: str,
        sample_texts: Dict[str, Any],
        golden_outputs: Dict[str, Any],
        legacy_preprocessor: TextPreprocessor,
    ):
        """Test legacy mode preprocessing by category."""
        golden_legacy = golden_outputs["outputs"]["legacy_mode"]

        # Skip if no golden outputs yet
        if not golden_legacy:
            pytest.skip("No golden outputs available. Run generate_golden_outputs() first.")

        # Filter test cases by category
        test_cases = [
            tc for tc in sample_texts["test_cases"]
            if tc.get("category") == category
        ]

        if not test_cases:
            pytest.skip(f"No test cases found for category: {category}")

        failures = []
        for test_case in test_cases:
            test_id = test_case["id"]
            input_text = test_case["input"]

            # Get expected output
            if test_id not in golden_legacy:
                failures.append(f"Test {test_id}: Missing from golden outputs")
                continue

            expected = golden_legacy[test_id]["output"]

            # Get actual output
            try:
                actual = legacy_preprocessor.preprocess(
                    input_text,
                    remove_stopwords=True,
                    lemmatize=True,
                    lowercase=True,
                    min_token_length=2,
                    track_metrics=False,
                )
            except TextPreprocessingError:
                actual = None

            # Compare
            match, message = compare_outputs(test_id, expected, actual)
            if not match:
                failures.append(message)

        assert not failures, "\n".join(failures)

    def test_legacy_mode_all_cases(
        self,
        sample_texts: Dict[str, Any],
        golden_outputs: Dict[str, Any],
        legacy_preprocessor: TextPreprocessor,
    ):
        """Test all sample texts in legacy mode."""
        golden_legacy = golden_outputs["outputs"]["legacy_mode"]

        # Skip if no golden outputs yet
        if not golden_legacy:
            pytest.skip("No golden outputs available. Run generate_golden_outputs() first.")

        failures = []
        passed = 0

        for test_case in sample_texts["test_cases"]:
            test_id = test_case["id"]
            input_text = test_case["input"]

            if test_id not in golden_legacy:
                failures.append(f"Test {test_id}: Missing from golden outputs")
                continue

            expected = golden_legacy[test_id]["output"]

            try:
                actual = legacy_preprocessor.preprocess(
                    input_text,
                    remove_stopwords=True,
                    lemmatize=True,
                    lowercase=True,
                    min_token_length=2,
                    track_metrics=False,
                )
            except TextPreprocessingError:
                actual = None

            match, message = compare_outputs(test_id, expected, actual)
            if match:
                passed += 1
            else:
                failures.append(message)

        total = len(sample_texts["test_cases"])
        print(f"\nLegacy mode: {passed}/{total} tests passed")

        assert not failures, f"\n{len(failures)} failures:\n" + "\n".join(failures)


# =============================================================================
# REGRESSION TESTS - ENHANCED MODE
# =============================================================================

class TestEnhancedModeRegression:
    """Regression tests for enhanced preprocessing mode."""

    def test_golden_outputs_exist(self, golden_outputs):
        """Verify golden outputs file has enhanced mode data."""
        assert "outputs" in golden_outputs, "Golden outputs missing 'outputs' key"
        assert "enhanced_mode" in golden_outputs["outputs"], "Golden outputs missing 'enhanced_mode'"
        assert len(golden_outputs["outputs"]["enhanced_mode"]) > 0, (
            "No enhanced mode outputs found. Run generate_golden_outputs() first."
        )

    @pytest.mark.parametrize("category", [
        "simple",
        "contractions",
        "negations",
        "urls_emails",
        "numbers",
        "domain_specific",
        "mixed_case",
        "special_characters",
        "empty_whitespace",
        "long_text",
        "social_media",
    ])
    def test_enhanced_mode_by_category(
        self,
        category: str,
        sample_texts: Dict[str, Any],
        golden_outputs: Dict[str, Any],
        enhanced_preprocessor: TextPreprocessor,
    ):
        """Test enhanced mode preprocessing by category."""
        golden_enhanced = golden_outputs["outputs"]["enhanced_mode"]

        # Skip if no golden outputs yet
        if not golden_enhanced:
            pytest.skip("No golden outputs available. Run generate_golden_outputs() first.")

        # Filter test cases by category
        test_cases = [
            tc for tc in sample_texts["test_cases"]
            if tc.get("category") == category
        ]

        if not test_cases:
            pytest.skip(f"No test cases found for category: {category}")

        failures = []
        for test_case in test_cases:
            test_id = test_case["id"]
            input_text = test_case["input"]

            if test_id not in golden_enhanced:
                failures.append(f"Test {test_id}: Missing from golden outputs")
                continue

            expected = golden_enhanced[test_id]["output"]

            try:
                actual = enhanced_preprocessor.preprocess(
                    input_text,
                    remove_stopwords=False,
                    lemmatize=False,
                    lowercase=True,
                    min_token_length=2,
                    track_metrics=False,
                )
            except TextPreprocessingError:
                actual = None

            match, message = compare_outputs(test_id, expected, actual)
            if not match:
                failures.append(message)

        assert not failures, "\n".join(failures)

    def test_enhanced_mode_all_cases(
        self,
        sample_texts: Dict[str, Any],
        golden_outputs: Dict[str, Any],
        enhanced_preprocessor: TextPreprocessor,
    ):
        """Test all sample texts in enhanced mode."""
        golden_enhanced = golden_outputs["outputs"]["enhanced_mode"]

        # Skip if no golden outputs yet
        if not golden_enhanced:
            pytest.skip("No golden outputs available. Run generate_golden_outputs() first.")

        failures = []
        passed = 0

        for test_case in sample_texts["test_cases"]:
            test_id = test_case["id"]
            input_text = test_case["input"]

            if test_id not in golden_enhanced:
                failures.append(f"Test {test_id}: Missing from golden outputs")
                continue

            expected = golden_enhanced[test_id]["output"]

            try:
                actual = enhanced_preprocessor.preprocess(
                    input_text,
                    remove_stopwords=False,
                    lemmatize=False,
                    lowercase=True,
                    min_token_length=2,
                    track_metrics=False,
                )
            except TextPreprocessingError:
                actual = None

            match, message = compare_outputs(test_id, expected, actual)
            if match:
                passed += 1
            else:
                failures.append(message)

        total = len(sample_texts["test_cases"])
        print(f"\nEnhanced mode: {passed}/{total} tests passed")

        assert not failures, f"\n{len(failures)} failures:\n" + "\n".join(failures)


# =============================================================================
# INDIVIDUAL TEST CASE TESTS
# =============================================================================

class TestIndividualCases:
    """Individual test cases for specific preprocessing behaviors."""

    def test_contraction_expansion_legacy(self, legacy_preprocessor, golden_outputs):
        """Test that contractions are properly expanded in legacy mode."""
        golden = golden_outputs["outputs"]["legacy_mode"]
        if not golden:
            pytest.skip("No golden outputs available")

        contraction_tests = [k for k in golden.keys() if k.startswith("contraction_")]
        for test_id in contraction_tests:
            expected = golden[test_id]["output"]
            input_text = golden[test_id]["input"]

            actual = legacy_preprocessor.preprocess(
                input_text,
                remove_stopwords=True,
                lemmatize=True,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )

            # Verify no contractions remain in output
            if actual:
                assert "'" not in actual or "n't" not in actual.lower(), (
                    f"Contraction still present in {test_id}: {actual}"
                )

            match, msg = compare_outputs(test_id, expected, actual)
            assert match, msg

    def test_url_standardization_legacy(self, legacy_preprocessor, golden_outputs):
        """Test that URLs are properly handled in legacy mode."""
        golden = golden_outputs["outputs"]["legacy_mode"]
        if not golden:
            pytest.skip("No golden outputs available")

        url_tests = [k for k in golden.keys() if k.startswith("url_email_")]
        for test_id in url_tests:
            expected = golden[test_id]["output"]
            input_text = golden[test_id]["input"]

            actual = legacy_preprocessor.preprocess(
                input_text,
                remove_stopwords=True,
                lemmatize=True,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )

            match, msg = compare_outputs(test_id, expected, actual)
            assert match, msg

    def test_empty_whitespace_handling(self, legacy_preprocessor, golden_outputs):
        """Test that empty/whitespace inputs are handled correctly."""
        golden = golden_outputs["outputs"]["legacy_mode"]
        if not golden:
            pytest.skip("No golden outputs available")

        empty_tests = [k for k in golden.keys() if k.startswith("empty_whitespace_")]
        for test_id in empty_tests:
            expected = golden[test_id]["output"]
            input_text = golden[test_id]["input"]

            actual = legacy_preprocessor.preprocess(
                input_text,
                remove_stopwords=True,
                lemmatize=True,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )

            match, msg = compare_outputs(test_id, expected, actual)
            assert match, msg

    def test_special_characters_normalization(self, legacy_preprocessor, golden_outputs):
        """Test that special characters are properly normalized."""
        golden = golden_outputs["outputs"]["legacy_mode"]
        if not golden:
            pytest.skip("No golden outputs available")

        special_tests = [k for k in golden.keys() if k.startswith("special_char_")]
        for test_id in special_tests:
            expected = golden[test_id]["output"]
            input_text = golden[test_id]["input"]

            actual = legacy_preprocessor.preprocess(
                input_text,
                remove_stopwords=True,
                lemmatize=True,
                lowercase=True,
                min_token_length=2,
                track_metrics=False,
            )

            # Verify no excessive punctuation remains
            if actual:
                assert "!!!" not in actual, f"Excessive punctuation in {test_id}: {actual}"
                assert "???" not in actual, f"Excessive punctuation in {test_id}: {actual}"

            match, msg = compare_outputs(test_id, expected, actual)
            assert match, msg


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestPreprocessingConsistency:
    """Tests to verify preprocessing consistency across runs."""

    def test_deterministic_output_legacy(self, legacy_preprocessor, sample_texts):
        """Verify legacy preprocessing produces deterministic output."""
        for test_case in sample_texts["test_cases"][:5]:  # Test first 5 cases
            input_text = test_case["input"]

            results = []
            for _ in range(3):
                try:
                    result = legacy_preprocessor.preprocess(
                        input_text,
                        remove_stopwords=True,
                        lemmatize=True,
                        lowercase=True,
                        min_token_length=2,
                        track_metrics=False,
                    )
                except TextPreprocessingError:
                    result = None
                results.append(result)

            assert all(r == results[0] for r in results), (
                f"Non-deterministic output for {test_case['id']}: {results}"
            )

    def test_deterministic_output_enhanced(self, enhanced_preprocessor, sample_texts):
        """Verify enhanced preprocessing produces deterministic output."""
        for test_case in sample_texts["test_cases"][:5]:  # Test first 5 cases
            input_text = test_case["input"]

            results = []
            for _ in range(3):
                try:
                    result = enhanced_preprocessor.preprocess(
                        input_text,
                        remove_stopwords=False,
                        lemmatize=False,
                        lowercase=True,
                        min_token_length=2,
                        track_metrics=False,
                    )
                except TextPreprocessingError:
                    result = None
                results.append(result)

            assert all(r == results[0] for r in results), (
                f"Non-deterministic output for {test_case['id']}: {results}"
            )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        print("Generating golden outputs...")
        generate_golden_outputs(force="--force" in sys.argv)
    else:
        print("Usage:")
        print("  python -m tests.test_preprocessing_regression --generate [--force]")
        print("  pytest tests/test_preprocessing_regression.py -v")
        print("  pytest tests/test_preprocessing_regression.py::test_generate_golden_outputs -v --generate-golden")
