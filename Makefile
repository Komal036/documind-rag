# ════════════════════════════════════════════════════════
#  DocuMind — Developer Command Reference
#  Run `make help` to see all available commands.
# ════════════════════════════════════════════════════════

.PHONY: help install install-dev lint format typecheck test test-unit \
        test-integration run-api run-frontend clean docker-build docker-up \
        docker-down ingest

PYTHON  := python3
PIP     := pip
VENV    := .venv
APP_DIR := src/api/main.py

## ── Setup ──────────────────────────────────────────────
help:                ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:             ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev:         ## Install all dependencies including dev/test tools
	$(PIP) install -r requirements-dev.txt
	pre-commit install

## ── Code Quality ───────────────────────────────────────
lint:                ## Run ruff linter
	ruff check src/ tests/

format:              ## Auto-format code with black + ruff
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:           ## Run mypy static type checker
	mypy src/

## ── Testing ─────────────────────────────────────────────
test:                ## Run all tests with coverage
	pytest tests/ --cov=src --cov-report=html

test-unit:           ## Run unit tests only
	pytest tests/unit/ -v

test-integration:    ## Run integration tests only
	pytest tests/integration/ -v

## ── Running Locally ─────────────────────────────────────
run-api:             ## Start the FastAPI backend
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:        ## Start the Streamlit frontend
	streamlit run frontend/streamlit_app.py --server.port 8501

## ── Ingestion ───────────────────────────────────────────
ingest:              ## Ingest documents from data/raw/
	$(PYTHON) -m src.ingestion.pipeline --source data/raw/

## ── Docker ──────────────────────────────────────────────
docker-build:        ## Build Docker images
	docker-compose -f deployment/docker/docker-compose.yml build

docker-up:           ## Start all services with Docker Compose
	docker-compose -f deployment/docker/docker-compose.yml up -d

docker-down:         ## Stop all Docker services
	docker-compose -f deployment/docker/docker-compose.yml down

## ── Cleanup ─────────────────────────────────────────────
clean:               ## Remove build artifacts, cache, logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ .ruff_cache/ .mypy_cache/
	rm -rf logs/*.log
	@echo "🧹 Cleaned up build artifacts"
