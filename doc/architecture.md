# Architecture Backend - GenoLens

## 1. Vue d'ensemble architecturale

Le backend GenoLens suit une architecture **layered (en couches)** avec une séparation claire entre les préoccupations :

```
┌─────────────────────────────────────────────────────┐
│                    Clients                          │
│   (Frontend Next.js, Mobile, Third-party)           │
└──────────────────┬──────────────────────────────────┘
                   │ HTTPS / REST API
┌──────────────────▼──────────────────────────────────┐
│              FastAPI Application                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Middleware   │  │ Routers/     │  │ Exception │  │
│  │ (CORS, Auth,│  │ Endpoints    │  │ Handlers  │  │
│  │ Rate Limit)  │  │              │  │           │  │
│  └─────────────┘  └──────────────┘  └───────────┘  │
┌──────────────────▼──────────────────────────────────┐
│              Services (Business Logic)               │
│  data_processor | go_service | gsea_processor       │
│  ai_interpreter | stripe_service | cache_service    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              Data Access Layer                       │
│  SQLAlchemy ORM + Alembic Migrations                │
└───────┬───────────────────────┬─────────────────────┘
        │                       │
┌───────▼──────┐      ┌────────▼────────┐
│ PostgreSQL   │      │  Redis Cache    │
│ (Metadata)   │      │  (TTL Caches)   │
└──────────────┘      └─────────────────┘
        │
┌───────▼──────┐
│ Supabase     │
│ Storage      │
│ (Fichiers    │
│  Parquet)    │
└──────────────┘
```

## 2. Patterns de conception utilisés

### 2.1 Dependency Injection (FastAPI Depends)

Les dépendances sont injectées via le système `Depends()` de FastAPI :

- **`get_db`** : Session de base de données async
- **`get_current_user`** : Utilisateur authentifié via Supabase JWT
- **`check_subscription`** : Vérification du plan d'abonnement

### 2.2 Service Layer Pattern

La logique métier est encapsulée dans des services réutilisables :

```python
# Exemple de service
class AnalysisService:
    @staticmethod
    async def run_dea_analysis(project_id, dataset_ids, ...):
        # Logique métier
        pass
    
    @staticmethod
    async def get_results(analysis_id):
        # Récupération des résultats
        pass
```

### 2.3 Schema Validation (Pydantic)

Toutes les entrées/sorties sont validées via Pydantic :

```python
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    species: Optional[str] = None
    
class ProjectResponse(ProjectCreate):
    id: UUID
    owner_id: UUID
    created_at: datetime
```

### 2.4 Repository-like Access via SQLAlchemy

Les modèles ORM sont utilisés directement avec des sessions async :

```python
async def get_project(db: AsyncSession, project_id: UUID):
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    return result.scalar_one_or_none()
```

## 3. Flux de données typique

### Upload d'un dataset

```
Client → POST /api/v1/datasets/upload
         │
         ├─► Validation (Pydantic schema)
         ├─► Auth check (Supabase JWT)
         ├─► Rate limit check
         │
         ├─► File processing (data_processor.py)
         │   ├─► Parsing CSV/TSV/XLSX
         │   ├─► Conversion Parquet
         │   └─► Upload Supabase Storage
         │
         ├─► Background task (Celery)
         │   └─► Stats computation, QC metrics
         │
         └─► Response: { dataset_id, status }
```

### Analyse différentielle

```
Client → POST /api/v1/analyses/differential-expression
         │
         ├─► Validation des paramètres
         ├─► Vérification quota utilisateur (Stripe)
         ├─► Enregistrement AnalysisRun (provenance)
         │
         ├─► Celery task: run_deseq_analysis()
         │   ├─► R script execution (DESeq2)
         │   ├─► Calcul des DEG
         │   └─► Sauvegarde résultats Parquet
         │
         └─► Polling / WebSocket → Résultats
```

## 4. Architecture de stockage hybride

### PostgreSQL (Métadonnées)

- Projets, datasets, utilisateurs
- Résultats d'analyses (références)
- Metadata des fichiers (chemins, formats)
- Abonnements Stripe
- Historique d'activité

### Fichiers Parquet (Données biologiques brutes)

- Matrices de comptage (genes × samples)
- Résultats DEG (log2FC, p-value, adj_pvalue)
- Résultats GSEA/GO enrichment
- Données de clustering/PCA/UMAP

**Raison du choix** : Les datasets transcriptomiques peuvent atteindre plusieurs centaines de MB. Parquet offre un compression efficace et des lectures rapides par colonnes.

### Supabase Storage (Fichiers uploadés)

- Fichiers sources CSV/TSV/XLSX
- Exportations utilisateur
- Données temporaires

## 5. Système de caching

```python
# Cache Redis distribué pour le scaling horizontal
cache_service = CacheService()
await cache_service.initialize(redis_url)

# Exemple d'utilisation
@cache_service.ttl_cache(ttl=3600, key_prefix="clustering")
async def get_clustering_result(dataset_id):
    # Calcul coûteux seulement si pas en cache
    return compute_clustering(dataset_id)
```

**Caches implémentés** :
- Résultats de clustering (Hierarchical, KMeans)
- Plots volcano (calculs statistiques)
- Stats des échantillons
- Computations GSEA

## 6. Worker Celery

### Configuration des queues

| Queue | Usage | Workers |
|---|---|---|
| `celery` | Tâches générales | Auto-scaled |
| `default` | Tâches par défaut | Auto-scaled |
| `data_processing` | Ingestion de datasets | 1-3 workers |
| `analysis` | Analyses bioinformatiques | 1-5 workers |

### Tasks principales

```python
# Ingestion de données
@app.task(bind=True, max_retries=3)
def ingest_dataset(self, dataset_id: str):
    """Parse et convertit un fichier uploadé en Parquet."""
    
# Analyses bioinformatiques  
@app.task(bind=True, max_retries=5)
def run_deseq_analysis(self, analysis_id: str):
    """Exécute DESeq2 via R pour l'analyse différentielle."""

@app.task(bind=True, max_retries=3)
def run_gsea_analysis(self, analysis_id: str):
    """Exécute GSEA avec les gene sets annotés."""

@app.task(bind=True, max_retries=3)
def compute_go_enrichment(self, analysis_id: str):
    """Calcule l'enrichissement GO (Fisher exact test)."""
```

## 7. Intégration IA (Ollama)

### Architecture

```
Client → POST /api/v1/ai/chat
         │
         ├─► Vérification quota AI
         ├─► Récupération contexte (données de l'analyse)
         ├─► Envoi à Ollama (modèle local)
         │   └─► llama3 / mistral / etc.
         │
         ├─► Sauvegarde conversation (DB)
         └─► Streaming response (SSE)
```

### Fonctionnalités IA

- **Interprétation de résultats** : Analyse des DEG, enrichment, plots
- **Assistant visualisation** : Suggestions de graphiques pertinents
- **Génération de code R/Python** : Pour analyses personnalisées
- **Résumé biologique** : Synthèse des résultats d'analyse

## 8. Monitoring & Observabilité

### Sentry (Erreurs)

```python
sentry_sdk.init(
    dsn=settings.sentry_dsn,
    environment=settings.sentry_environment,
    traces_sample_rate=0.1,
)
```

### Prometheus (Métriques)

Endpoint `/metrics` exposé automatiquement par `prometheus-fastapi-instrumentator` :

- HTTP request rates & latencies
- Database connection pool stats
- Custom business metrics

### Grafana (Dashboards)

Dashboard fourni dans `monitoring/grafana/provisioning/dashboards/genolens-api.json`.

## 9. Sécurité

| Couche | Mécanisme |
|---|---|
| Auth | Supabase JWT validation |
| Rate Limiting | slowapi (configurable par endpoint) |
| CORS | Whitelist d'origines via config |
| Headers | SecurityHeadersMiddleware (HSTS, CSP, X-Frame-Options...) |
| Validation | Pydantic schemas stricts sur toutes les entrées |
| Storage | Supabase RLS policies pour l'accès aux fichiers |

## 10. Scalabilité

### Horizontal

- Stateless API servers (multi-instance derrière load balancer)
- Redis distribué pour le caching
- PostgreSQL avec connection pooling (PgBouncer recommandé en prod)

### Vertical

- Chunked processing pour les gros fichiers (`CHUNK_SIZE` config)
- Celery workers auto-scaled selon la file d'attente
- Parquet compression ajustable (`PARQUET_COMPRESSION`)

## 11. Cycle de vie de l'application

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await cache_service.initialize(settings.REDIS_URL)
    
    yield
    
    # Shutdown
    await cache_service.close()
    await close_db()
```

Les événements de startup/shutdown garantissent une initialization propre des ressources partagées.