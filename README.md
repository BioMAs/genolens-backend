# GenoLens Next

A flexible, data-centric bioinformatics SaaS platform for transcriptomics analysis with a **Bring Your Own Data (BYOD)** approach.

## Overview

GenoLens Next allows researchers to upload processed transcriptomics data (count matrices, differential expression results, enrichment tables) and query them on-demand through a high-performance API. Unlike traditional pipelines, this platform focuses on flexible data storage and retrieval.

### Key Features

- **Asset-Based Architecture**: Metadata in PostgreSQL, data in Parquet files
- **Lazy Loading**: Query data on-demand with filters (genes, samples)
- **Background Processing**: Async conversion of CSV/Excel to optimized Parquet
- **Supabase Integration**: Authentication and S3-compatible storage
- **High Performance**: FastAPI with async SQLAlchemy
- **Scalable**: Celery task queue with Redis

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   FastAPI    │─────▶│ PostgreSQL  │
│   (React)   │      │   (Async)    │      │  (Metadata) │
└─────────────┘      └──────────────┘      └─────────────┘
                             │
                             │ triggers
                             ▼
                     ┌──────────────┐      ┌─────────────┐
                     │    Celery    │─────▶│  Supabase   │
                     │    Worker    │      │   Storage   │
                     └──────────────┘      │  (.parquet) │
                             │              └─────────────┘
                             ▼
                     ┌──────────────┐
                     │    Redis     │
                     │   (Broker)   │
                     └──────────────┘
```

### Data Flow

1. **Upload**: User uploads CSV/Excel via API
2. **Store Raw**: File saved to Supabase Storage
3. **Process**: Celery task converts to Parquet
4. **Query**: API reads Parquet on-demand with filters

## Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL (via Supabase or local)
- **ORM**: SQLAlchemy (Async) + Alembic
- **Authentication**: Supabase Auth (JWT)
- **Task Queue**: Celery + Redis
- **Data Processing**: Pandas + PyArrow
- **Storage**: Supabase Storage (S3-compatible)
- **Containerization**: Docker + Docker Compose

## Project Structure

```
genolens_v2/
├── app/
│   ├── api/
│   │   ├── deps/           # Dependencies (auth, db)
│   │   └── endpoints/      # API routes
│   │       ├── projects.py
│   │       └── datasets.py
│   ├── core/
│   │   ├── config.py       # Pydantic settings
│   │   └── security.py     # JWT validation
│   ├── db/
│   │   └── session.py      # Database connection
│   ├── models/
│   │   ├── base.py         # SQLAlchemy base
│   │   └── models.py       # Project, Sample, Dataset
│   ├── schemas/
│   │   ├── project.py      # Pydantic schemas
│   │   └── dataset.py
│   ├── services/
│   │   ├── storage.py      # Supabase Storage client
│   │   └── data_processor.py  # Parquet conversion
│   ├── worker/
│   │   ├── celery_app.py   # Celery config
│   │   └── tasks.py        # Background tasks
│   └── main.py             # FastAPI app
├── alembic/
│   ├── versions/           # Migration files
│   └── env.py              # Alembic config
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Supabase account (for production)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd genolens_v2
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your Supabase credentials:

```env
SUPABASE_URL=https://[YOUR-PROJECT-REF].supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_JWT_SECRET=your-jwt-secret-here
SUPABASE_STORAGE_BUCKET=genolens-data
```

Get these values from:
- Supabase Dashboard → Settings → API
- Create a storage bucket named `genolens-data`

### 3. Start with Docker Compose

```bash
# Start all services (API, Worker, Redis, PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f api
```

Services will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Flower (Celery Monitor): http://localhost:5555

### 4. Run Database Migrations

```bash
# Create initial migration
docker-compose exec api alembic revision --autogenerate -m "Initial migration"

# Apply migrations
docker-compose exec api alembic upgrade head
```

### 5. Test the API

Visit http://localhost:8000/docs for interactive API documentation.

## Development Setup (Without Docker)

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Start Services

You'll need to run these in separate terminals:

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API
uvicorn app.main:app --reload --port 8000

# Terminal 3: Start Celery Worker
celery -A app.worker.celery_app worker --loglevel=info
```

## API Endpoints

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/projects/` | Create project |
| GET | `/api/v1/projects/` | List projects |
| GET | `/api/v1/projects/{id}` | Get project |
| PATCH | `/api/v1/projects/{id}` | Update project |
| DELETE | `/api/v1/projects/{id}` | Delete project |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/datasets/upload` | Upload dataset file |
| GET | `/api/v1/datasets/{id}` | Get dataset metadata |
| GET | `/api/v1/datasets/{id}/query` | Query dataset data |
| GET | `/api/v1/datasets/project/{project_id}` | List project datasets |

### Example: Upload Dataset

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "project_id=YOUR_PROJECT_ID" \
  -F "name=Count Matrix" \
  -F "dataset_type=MATRIX" \
  -F "file=@counts.csv"
```

### Example: Query Dataset

```bash
curl "http://localhost:8000/api/v1/datasets/{dataset_id}/query?gene_ids=ENSG00000001&limit=100" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Database Models

### Project
- Organizes related samples and datasets
- Owned by a Supabase Auth user

### Sample
- Represents a biological sample
- Stores flexible metadata (condition, treatment, etc.)

### Dataset
- References data files in Supabase Storage
- Types: MATRIX, DEG, ENRICHMENT, METADATA
- Statuses: PENDING → PROCESSING → READY | FAILED

## Background Tasks

### `process_dataset_upload`

1. Downloads raw file from storage
2. Converts CSV/Excel to Parquet
3. Extracts metadata (rows, columns, dtypes)
4. Uploads Parquet to storage
5. Updates database status

## Configuration

All configuration is in [app/core/config.py](app/core/config.py) using Pydantic Settings.

Key settings:
- `DATABASE_URL`: PostgreSQL connection
- `REDIS_URL`: Redis connection
- `SUPABASE_*`: Supabase credentials
- `MAX_UPLOAD_SIZE`: File size limit (default: 500MB)
- `PARQUET_COMPRESSION`: Compression algorithm (default: snappy)

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## Production Deployment

### Supabase Setup

1. Create a Supabase project
2. Create a storage bucket: `genolens-data`
3. Set bucket to private
4. Get connection string from Dashboard → Settings → Database
5. Update `DATABASE_URL` to Supabase PostgreSQL

### Environment Variables

Update `.env` for production:

```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
```

### Deploy with Docker

```bash
docker-compose up -d --build
```

## Monitoring

- **Flower**: Monitor Celery tasks at http://localhost:5555
- **FastAPI Docs**: Interactive API docs at `/docs`
- **Health Check**: `GET /health`

## Security

- JWT validation via Supabase Auth
- Row-level security via `owner_id` checks
- File upload validation (extension, size)
- CORS configuration

## Performance Considerations

- **Parquet**: Columnar format for efficient querying
- **Lazy Loading**: Data loaded on-demand, not in PostgreSQL
- **Async I/O**: Non-blocking database and HTTP operations
- **Connection Pooling**: Configured in SQLAlchemy
- **Task Queues**: Background processing doesn't block API

## Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps

# View database logs
docker-compose logs db
```

### Celery Tasks Not Processing

```bash
# Check worker logs
docker-compose logs worker

# Verify Redis connection
redis-cli ping
```

### Alembic Migration Conflicts

```bash
# Reset migrations (CAUTION: drops data)
docker-compose exec api alembic downgrade base
docker-compose exec api alembic upgrade head
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- GitHub Issues: [Repository Issues]
- Email: support@genolens.com

---

**Built with ❤️ for the bioinformatics community**
