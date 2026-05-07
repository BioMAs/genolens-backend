# Déploiement Backend - GenoLens

## Vue d'ensemble

Le backend est déployé via Docker avec deux configurations : développement et production.

---

## 1. Environnement de Développement

### docker-compose.yml

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app        # Hot reload du code
      - ./data:/app/data      # Storage persistant

  worker:
    build: .
    command: >
      celery -A app.worker.celery_app worker --loglevel=info
      -Q data_processing,analysis
    env_file: .env
    depends_on:
      - redis
      - db
    volumes:
      - ./app:/app/app

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: genolens
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  pgdata:
  redis_data:
  ollama_data:
```

### Démarrage

```bash
# Construire et démarrer tous les services
docker-compose up --build

# En mode detached
docker-compose up -d

# Arrêter
docker-compose down

# Arrêter + supprimer les volumes (données perdues)
docker-compose down -v
```

### Migrations

```bash
# Dans le conteneur API
docker-compose exec api alembic upgrade head

# Créer une migration
docker-compose exec api alembic revision --autogenerate -m "description"
```

---

## 2. Environnement de Production

### docker-compose.prod.yml

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@db:5432/genolens_prod
      - REDIS_URL=redis://redis:6379/0
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - SUPABASE_SERVICE_ROLE_KEY=${SUPABASE_SERVICE_ROLE_KEY}
      - SUPABASE_JWT_SECRET=${SUPABASE_JWT_SECRET}
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
      - SENTRY_DSN=${SENTRY_DSN}
    depends_on:
      - db
      - redis
    labels:
      - "traefik.http.routers.genolens-api.rule=Host(`api.genolens.com`)"
      - "traefik.http.routers.genolens-api.tls=true"
      - "traefik.http.routers.genolens-api.tls.certresolver=letsencrypt"

  worker:
    build: .
    command: >
      celery -A app.worker.celery_app worker --loglevel=info
      -Q data_processing,analysis
      --concurrency=4
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@db:5432/genolens_prod
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - db
    deploy:
      replicas: 3

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: genolens_prod
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  redis_data:
```

### Démarrage production

```bash
# Copier le .env de production
cp .env.production .env

# Démarrer
docker-compose -f docker-compose.prod.yml up -d

# Vérifier les logs
docker-compose -f docker-compose.prod.yml logs -f api
docker-compose -f docker-compose.prod.yml logs -f worker
```

---

## 3. Variables d'environnement

### .env (Développement)

```bash
# Application
APP_NAME=GenoLens Next
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/genolens

# Redis
REDIS_URL=redis://redis:6379/0

# Supabase (local ou cloud)
SUPABASE_URL=http://host.docker.internal:54321
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret
SUPABASE_STORAGE_BUCKET=genolens-data

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Storage
LOCAL_STORAGE_PATH=/app/data
MAX_UPLOAD_SIZE=524288000  # 500 MB

# Data processing
PARQUET_COMPRESSION=snappy
CHUNK_SIZE=10000

# Frontend URL (pour les emails)
APP_URL=http://localhost:3000

# Stripe (test mode)
stripe_secret_key=sk_test_...
stripe_publishable_key=pk_test_...
stripe_webhook_secret=whsec_...
stripe_price_premium_monthly=price_xxx
stripe_price_advanced_monthly=price_yyy

# Email / SMTP
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=your-sendgrid-key
EMAIL_FROM_ADDRESS=noreply@genolens.com
EMAIL_FROM_NAME=GenoLens

# Monitoring (optionnel)
sentry_dsn=
sentry_environment=development
sentry_traces_sample_rate=0.1
```

### .env.production

```bash
ENVIRONMENT=production
DEBUG=false

DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@db:5432/genolens_prod

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret

CORS_ORIGINS=["https://app.genolens.com"]

stripe_secret_key=sk_live_...
stripe_publishable_key=pk_live_...
stripe_webhook_secret=whsec_...

sentry_dsn=https://xxx@o0.ingest.sentry.io/0
sentry_environment=production
sentry_traces_sample_rate=0.1

APP_URL=https://app.genolens.com
```

---

## 4. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 5. Monitoring & Observabilité

### Sentry (Erreurs)

```python
# app/main.py
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
    )
```

### Prometheus (Métriques)

Endpoint `/metrics` exposé automatiquement :

```bash
# Tester localement
curl http://localhost:8000/metrics
```

Métriques principales :
- `http_request_duration_seconds` — Latence des requêtes HTTP
- `http_requests_total` — Nombre de requêtes par endpoint/status
- `celery_task_success_total` / `celery_task_failure_total` — Statut des tâches Celery

### Grafana Dashboard

Le dashboard est provisionné automatiquement via :
```
monitoring/grafana/provisioning/dashboards/dashboard.yml
monitoring/grafana/provisioning/dashboards/genolens-api.json
```

---

## 6. Backup & Restore

### Script de backup (`backup.sh`)

```bash
#!/bin/bash
# Backup PostgreSQL database

DB_NAME="genolens"
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

docker exec genolens-db pg_dump -U postgres "$DB_NAME" | gzip > "$BACKUP_FILE"

# Rotation : garder les 7 derniers backups
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_FILE"
```

### Restore

```bash
# Restaurer un backup
gunzip < /backups/genolens_20260101_120000.sql.gz | \
    docker exec -i genolens-db psql -U postgres genolens
```

---

## 7. CI/CD (Recommandations)

### Pipeline GitHub Actions

```yaml
# .github/workflows/backend-ci.yml
name: Backend CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: genolens_test
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run migrations
        run: alembic upgrade head
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/genolens_test
      
      - name: Run tests
        run: pytest -v --cov=app --cov-report=xml
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/genolens_test
          REDIS_URL: redis://localhost:6379/0
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 8. Déploiement manuel (VPS)

### Prérequis serveur

- Ubuntu 22.04+
- Docker & Docker Compose installé
- PostgreSQL 16+ (ou utiliser le conteneur)
- Redis 7+ (ou utiliser le conteneur)
- Nginx en reverse proxy

### Étapes

```bash
# 1. Cloner le repo
git clone https://github.com/genolens/genolens.git
cd genolens/backend

# 2. Configurer les variables d'environnement
cp .env.production .env
nano .env  # Modifier les valeurs sensibles

# 3. Construire et démarrer
docker-compose -f docker-compose.prod.yml up -d --build

# 4. Appliquer les migrations
docker-compose exec api alembic upgrade head

# 5. Configurer Nginx
sudo nano /etc/nginx/sites-available/genolens-api

# Exemple de config Nginx :
server {
    listen 80;
    server_name api.genolens.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# 6. Activer et recharger Nginx
sudo ln -s /etc/nginx/sites-available/genolens-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 7. Configurer SSL avec Let's Encrypt
sudo certbot --nginx -d api.genolens.com
```

---

## 9. Scaling horizontal

### Multi-instance API

```yaml
# docker-compose.prod.yml (extrait)
services:
  api:
    deploy:
      replicas: 3
    # ... autres config
```

### Load balancing

Derrière un load balancer (Traefik, Nginx, HAProxy) :

```yaml
# Traefik configuration
labels:
  - "traefik.http.services.genolens-api.loadbalancer.server.port=8000"
  - "traefik.http.routers.genolens-api.rule=Host(`api.genolens.com`)"
```

### Worker scaling

```yaml
worker:
  deploy:
    replicas: 5
  command: >
    celery -A app.worker.celery_app worker --loglevel=info
    -Q data_processing,analysis
    --concurrency=4
```

---

## 10. Troubleshooting

### Problèmes courants

| Problème | Solution |
|---|---|
| `connection refused` sur la DB | Vérifier que le service db est up : `docker-compose ps` |
| Migrations échouent | Vérifier le DATABASE_URL et les permissions |
| Worker ne démarre pas | Vérifier Redis : `docker exec -it <redis_container> redis-cli ping` |
| CORS errors | Ajouter l'origine frontend dans CORS_ORIGINS |
| Ollama non disponible | Vérifier que le service ollama est accessible sur port 11434 |

### Logs de debug

```bash
# Tous les logs
docker-compose -f docker-compose.prod.yml logs

# Logs spécifiques
docker-compose -f docker-compose.prod.yml logs -f api
docker-compose -f docker-compose.prod.yml logs -f worker

# Shell dans un conteneur
docker-compose exec api bash
docker-compose exec db psql -U postgres genolens_prod
```

### Health checks

```bash
# API health
curl http://localhost:8000/health

# DB connectivity
curl http://localhost:8000/db-test

# Redis ping
docker exec <redis_container> redis-cli ping
```