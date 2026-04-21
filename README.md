# Genolens Backend

Backend API for the Genolens bioinformatics platform, built with FastAPI, SQLAlchemy, and Celery.

## 🚀 Features

- **High Performance API**: Built with FastAPI and async Python.
- **Data Processing**: Specialized parsing for bio-data (CSV/TSV/Parquet) for DEG, Enrichment, and Counts.
- **Asynchronous Tasks**: Celery workers backed by Redis for heavy data ingestion and processing.
- **AI Integration**: Integration with Ollama for localized LLM biological interpretation.
- **Security**: Supabase Auth integration.
- **Storage**: Hybrid storage approach using PostgreSQL for metadata/relational data and Parquet files for large biological datasets.
- **Payments**: Stripe subscription management (Checkout, Customer Portal, webhook plan enforcement) — see [STRIPE_BILLING.md](../docs/STRIPE_BILLING.md).
- **Monitoring**: Sentry error tracking, Prometheus `/metrics` endpoint, Grafana dashboard — see [MONITORING.md](../docs/MONITORING.md).
- **Shared Cache**: Redis-backed async TTL caches (clustering, volcano, stats) replace in-memory TTLCache for horizontal scaling.
- **Analysis Provenance**: Every GSEA, GO enrichment, clustering, and volcano computation saves an `AnalysisRun` record with parameters and package versions — see [ANALYSIS_PROVENANCE.md](../docs/ANALYSIS_PROVENANCE.md).

## 🛠 Prerequisites

- **Docker** and **Docker Compose**
- **Python 3.10+** (if running locally without Docker)
- **Supabase** account (or local instance) for Auth and Storage buckets.

## 📦 Installation & Setup

### 1. Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

You need to fill in at least:
- `DATABASE_URL` (PostgreSQL connection string)
- `SUPABASE_URL` and keys
- `REDIS_URL`

### 2. Running with Docker (Recommended)

To start the API, Worker, Database, Redis, and Ollama:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.
API Documentation (Swagger UI): `http://localhost:8000/docs`.

### 3. Running Locally (Development)

Ensure you have a PostgreSQL database and Redis running locally (or via docker-compose).

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload
```

To run the worker locally:
```bash
celery -A app.worker.celery_app worker --loglevel=info -Q celery,default,data_processing
```

## 📂 Project Structure

```
backend/
├── alembic/              # Database migrations
├── app/
│   ├── api/              # API endpoints (Routes)
│   ├── core/             # Config, Security, Auth
│   ├── db/               # Database session and base models
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas (Request/Response)
│   ├── services/         # Business logic (Data processing, AI, stats)
│   └── worker/           # Celery task definitions
├── sql/                  # Raw SQL schemas (reference)
├── scripts/              # Helper scripts (setup, ingestion, testing)
└── tests/                # Pytest tests
```

## 🧪 Testing

```bash
pytest
```

## 🚢 Production

Use the provided `docker-compose.prod.yml` for production deployments. It includes Traefik labels for reverse proxy configuration.

```bash
docker-compose -f docker-compose.prod.yml up -d
```
