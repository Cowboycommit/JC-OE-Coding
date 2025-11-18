.PHONY: help install test lint format typecheck quality clean docs test-coverage

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
PYLINT := pylint
MYPY := mypy
ISORT := isort

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
OUTPUT_DIR := output

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run unit tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make lint           - Run code linting (flake8, pylint)"
	@echo "  make format         - Format code with black and isort"
	@echo "  make typecheck      - Run type checking with mypy"
	@echo "  make quality        - Run all quality checks"
	@echo "  make clean          - Clean up generated files"
	@echo "  make docs           - Generate documentation"
	@echo "  make notebook       - Launch Jupyter notebook"
	@echo "  make all            - Install, format, lint, and test"

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

test:
	@echo "Running tests..."
	$(PYTEST) $(TEST_DIR) -v --tb=short
	@echo "✓ Tests completed"

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-all: test test-coverage
	@echo "✓ All tests completed"

lint:
	@echo "Running flake8..."
	$(FLAKE8) $(SRC_DIR) --max-line-length=100 --extend-ignore=E203,W503
	@echo "Running pylint..."
	$(PYLINT) $(SRC_DIR) --max-line-length=100 --disable=C0111,R0903
	@echo "✓ Linting completed"

format:
	@echo "Formatting code with black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=100
	@echo "Sorting imports with isort..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR) --profile black
	@echo "✓ Code formatted"

typecheck:
	@echo "Running type checks..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --no-strict-optional
	@echo "✓ Type checking completed"

quality: format lint typecheck
	@echo "✓ All quality checks passed"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "✓ Cleanup completed"

docs:
	@echo "Generating documentation..."
	@mkdir -p $(DOCS_DIR)
	$(PYTHON) -m pydoc -w $(SRC_DIR)/*.py
	@mv *.html $(DOCS_DIR)/ 2>/dev/null || true
	@echo "✓ Documentation generated in $(DOCS_DIR)/"

notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook open_ended_coding_analysis.ipynb

setup-dirs:
	@echo "Creating project directories..."
	@mkdir -p $(SRC_DIR)
	@mkdir -p $(TEST_DIR)
	@mkdir -p data
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(DOCS_DIR)
	@touch $(SRC_DIR)/__init__.py
	@touch $(TEST_DIR)/__init__.py
	@echo "✓ Project directories created"

all: install format lint typecheck test
	@echo "✓ All tasks completed successfully"

# Development workflow
dev: format lint test
	@echo "✓ Development checks passed"

# CI/CD workflow
ci: install quality test-coverage
	@echo "✓ CI checks completed"
