.PHONY: install test typecheck lint fmt clean build publish all check test-verbose

# Use python3 if available, otherwise fall back to python
PYTHON := $(shell command -v python3 || command -v python)

# Default target
all: install typecheck fmt lint test

# Run all checks
check: test typecheck lint


# Install dependencies
install:
	$(PYTHON) -m pip install -e .[dev]

# Run tests
test:
	$(PYTHON) -m pytest -n auto

# Run tests
test-verbose:
	$(PYTHON) -m pytest

# Run type checking
typecheck:
	$(PYTHON) -m mypy .

# Run linter
lint:
	$(PYTHON) -m ruff check .

# Format code
fmt:
	$(PYTHON) -m ruff check . --fix

# Clean up build artifacts
clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

# Build the project
build:
	flit build

publish:
	flit publish


