"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest before running any tests.
"""

import os

# Fix Keras 3 compatibility issue with HuggingFace Transformers
# Must be set BEFORE importing tensorflow or transformers
# Keras 3 is not yet supported by transformers, so we configure TensorFlow
# to use the backwards-compatible tf-keras package instead.
# See: https://github.com/huggingface/transformers/issues/27850
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--generate-golden",
        action="store_true",
        default=False,
        help="Generate golden outputs for preprocessing regression tests",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "generate_golden: mark test as golden output generator"
    )


@pytest.fixture(scope="session", autouse=True)
def keras_compatibility_setup():
    """Ensure Keras compatibility environment variable is set for all tests."""
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    yield
