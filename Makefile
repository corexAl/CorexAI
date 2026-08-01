.PHONY: help install dev test lint format clean run docker-build docker-run

PYTHON=python3
PIP=pip3

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make dev           Install development dependencies"
	@echo "  make run           Run the project"
	@echo "  make test          Run tests"
	@echo "  make lint          Run Ruff"
	@echo "  make format        Format code with Black"
	@echo "  make clean         Remove cache files"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

run:
	$(PYTHON) main.py

test:
	pytest

lint:
	ruff check .

format:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

docker-build:
	docker build -t my-llm .

docker-run:
	docker run -it --rm my-llm
