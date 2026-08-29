# Makefile for AI Grooming Assistant

.PHONY: help setup install run test lint format clean dev-setup

# Default target
help:
	@echo "AI Grooming Assistant - Available commands:"
	@echo ""
	@echo "  make setup          - Initial project setup"
	@echo "  make install        - Install dependencies"
	@echo "  make run            - Run the application"
	@echo "  make dev-setup      - Setup development environment"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make requirements   - Generate requirements.txt"
	@echo ""

# Setup
setup:
	@echo "Setting up AI Grooming Assistant..."
	@python setup.py

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Installation complete"

# Dev setup
dev-setup:
	@echo "Setting up development environment..."
	@pip install -r requirements-dev.txt
	@echo "✓ Development environment ready"

# Run application
run:
	@echo "Starting AI Grooming Assistant..."
	@python run.py

# Run with specific port
run-port:
	@read -p "Enter port number [5000]: " port; \
	port=$${port:-5000}; \
	python -c "from app import app; app.run(debug=True, host='0.0.0.0', port=$$port)"

# Testing
test:
	@echo "Running tests..."
	@pytest tests/ -v --cov=. --cov-report=html
	@echo "✓ Tests complete. Coverage report: htmlcov/index.html"

# Linting
lint:
	@echo "Running linting..."
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@pylint app.py predict.py config.py detectors/ --disable=all --enable=E,F
	@echo "✓ Linting complete"

# Code formatting
format:
	@echo "Formatting code..."
	@black .
	@isort .
	@echo "✓ Code formatted"

# Clean temporary files
clean:
	@echo "Cleaning up..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	@echo "✓ Cleanup complete"

# Generate requirements
requirements:
	@echo "Updating requirements.txt..."
	@pip freeze > requirements.txt
	@echo "✓ requirements.txt updated"

# Kill process on port 5000
kill-port:
	@echo "Killing process on port 5000..."
	@lsof -ti:5000 | xargs kill -9 2>/dev/null || echo "No process on port 5000"

# Check Python syntax
check:
	@echo "Checking Python syntax..."
	@python -m py_compile app.py predict.py config.py
	@python -m py_compile detectors/*.py
	@echo "✓ All files have valid syntax"

# Install pre-commit hooks
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	@pip install pre-commit
	@pre-commit install
	@echo "✓ Pre-commit hooks installed"
