# SumUp Escalation Detection System Makefile

.PHONY: help install test lint format clean build run docker-build docker-run deploy

# Default target
help:
	@echo "SumUp Escalation Detection System"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the application"
	@echo "  run          Run the API service"
	@echo "  cli          Run the CLI interface"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo "  deploy       Deploy to Kubernetes"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock flake8 mypy bandit safety

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/test_*.py -v -m "not integration"

test-integration:
	pytest tests/test_integration.py -v

# Code quality
lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# Security
security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build
build: clean
	python -m build

# Run services
run:
	python -m src.service

cli:
	python cli/main.py

# Docker
docker-build:
	docker build -t sumup/escalation-detector:latest .

docker-run:
	docker-compose up --build

docker-stop:
	docker-compose down

# Kubernetes deployment
deploy:
	kubectl apply -f k8s/deployment.yaml

deploy-dev:
	kubectl apply -f k8s/deployment.yaml -n development

deploy-prod:
	kubectl apply -f k8s/deployment.yaml -n production

# Development
dev-setup: install
	@echo "Setting up development environment..."
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Activating virtual environment..."
	@echo "Run: source venv/bin/activate"
	@echo "Then run: make install"

# Monitoring
logs:
	docker-compose logs -f escalation-api

health:
	curl -f http://localhost:8080/health

metrics:
	curl -f http://localhost:8080/metrics

# Data processing
train:
	@echo "Running training notebook..."
	jupyter nbconvert --execute notebooks/escalation_detector.ipynb --to notebook --output-dir=notebooks/

# Documentation
docs:
	@echo "Generating API documentation..."
	python -m src.service &
	sleep 5
	curl -f http://localhost:8080/docs
	pkill -f "python -m src.service"

# All-in-one commands
ci: install test lint security
	@echo "CI pipeline completed successfully"

release: clean test lint security docker-build
	@echo "Release build completed successfully"
