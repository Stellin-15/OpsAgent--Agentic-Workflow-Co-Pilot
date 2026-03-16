.PHONY: dev test test-unit test-integration test-e2e migrate migrate-create lint typecheck clean index-runbooks help

# ── Development ───────────────────────────────────────────────────────────────

dev:
	docker compose up --build

dev-db:
	docker compose up postgres -d

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --cov=opsagent --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Database ──────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

migrate-create:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

test-e2e:
	cd frontend && npm run test:e2e

# ── Runbook indexing (Phase 4) ─────────────────────────────────────────────────

index-runbooks:
	python scripts/index_runbooks.py

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check opsagent/ tests/
	ruff format --check opsagent/ tests/

lint-fix:
	ruff check --fix opsagent/ tests/
	ruff format opsagent/ tests/

typecheck:
	mypy opsagent/

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

help:
	@echo "Available targets:"
	@echo "  dev              - Start all services with docker compose"
	@echo "  dev-db           - Start only the database"
	@echo "  test             - Run full test suite with coverage"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  migrate          - Apply all pending Alembic migrations"
	@echo "  migrate-down     - Rollback one migration"
	@echo "  migrate-create   - Create a new migration (autogenerate)"
	@echo "  lint             - Check code style"
	@echo "  lint-fix         - Auto-fix code style issues"
	@echo "  typecheck        - Run mypy type checking"
	@echo "  test-e2e         - Run Playwright e2e tests"
	@echo "  index-runbooks   - Manually rebuild pgvector runbook index"
	@echo "  clean            - Remove build artifacts and caches"
