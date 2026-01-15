# GenoLens Next - Makefile for common commands

.PHONY: help build up down logs shell test migrate migration clean setup verify test-api

help:
	@echo "GenoLens Next - Available Commands"
	@echo "==================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "make setup        - Complete setup (start + migrate)"
	@echo "make verify       - Verify setup is working"
	@echo "make test-api     - Run API tests"
	@echo ""
	@echo "🐳 Docker:"
	@echo "make build        - Build Docker images"
	@echo "make up           - Start all services"
	@echo "make down         - Stop all services"
	@echo "make restart      - Restart all services"
	@echo ""
	@echo "📝 Logs:"
	@echo "make logs         - View logs (all services)"
	@echo "make logs-api     - View API logs"
	@echo "make logs-worker  - View worker logs"
	@echo ""
	@echo "🗄️ Database:"
	@echo "make migration    - Create new migration"
	@echo "make migrate      - Apply migrations"
	@echo "make db-shell     - Open database shell"
	@echo ""
	@echo "🔧 Development:"
	@echo "make shell        - Open API container shell"
	@echo "make test         - Run tests"
	@echo "make format       - Format code with black & isort"
	@echo "make lint         - Run code linting"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "make clean        - Remove containers and volumes"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Flower: http://localhost:5555"

down:
	docker-compose down

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-worker:
	docker-compose logs -f worker

shell:
	docker-compose exec api /bin/bash

migration:
	@read -p "Enter migration message: " msg; \
	docker-compose exec api alembic revision --autogenerate -m "$$msg"

migrate:
	docker-compose exec api alembic upgrade head

test:
	docker-compose exec api pytest

clean:
	docker-compose down -v
	docker system prune -f

format:
	black app/
	isort app/

lint:
	flake8 app/
	mypy app/

# Quick Start Commands
setup:
	@echo "🚀 Starting GenoLens Next setup..."
	./scripts/quick_setup.sh

verify:
	@echo "🔍 Verifying setup..."
	./scripts/verify_setup.sh

test-api:
	@echo "🧪 Testing API..."
	./scripts/test_api.sh

restart:
	docker-compose restart
	@echo "♻️ Services restarted"

db-shell:
	docker-compose exec db psql -U genolens -d genolens_db
