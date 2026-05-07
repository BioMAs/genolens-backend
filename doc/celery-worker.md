# Worker Celery - GenoLens Backend

## Vue d'ensemble

Celery est utilisé pour exécuter les tâches asynchrones lourdes (processing de fichiers, analyses bioinformatiques) sans bloquer l'API.

### Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   FastAPI    │────►│   Redis     │────►│   Celery     │
│   (Producer) │     │  (Broker)   │     │   Workers    │
└──────────────┘     └─────────────┘     └──────────────┘
                              │                    │
                              ▼                    ▼
                       ┌─────────────┐     ┌──────────────┐
                       │  Redis      │     │   R / Python │
                       │ (Results)   │     │   Processes  │
                       └─────────────┘     └──────────────┘
```

---

## Configuration

### Initialisation (`app/worker/celery_app.py`)

```python
from celery import Celery

celery_app = Celery(
    "genolens",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    
    # Queues
    task_queues=(
        Queue("celery"),
        Queue("default"),
        Queue("data_processing"),
        Queue("analysis"),
    ),
    
    # Routing
    task_routes={
        "app.worker.tasks.ingest_dataset": {"queue": "data_processing"},
        "app.worker.tasks.run_deseq_analysis": {"queue": "analysis"},
        "app.worker.tasks.run_gsea_analysis": {"queue": "analysis"},
        "app.worker.tasks.compute_go_enrichment": {"queue": "analysis"},
    },
    
    # Retry configuration
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
```

### Démarrage des workers

```bash
# Worker principal (toutes les queues)
celery -A app.worker.celery_app worker --loglevel=info

# Worker spécifique à la data processing
celery -A app.worker.celery_app worker --loglevel=info -Q data_processing

# Worker pour les analyses bioinformatiques
celery -A app.worker.celery_app worker --loglevel=info -Q analysis

# En production (multiple workers)
docker-compose up -d worker
```

---

## Tâches principales

### 1. Ingestion de datasets

```python
@app.task(bind=True, max_retries=3, queue="data_processing")
def ingest_dataset(self, dataset_id: str):
    """
    Parse un fichier uploadé (CSV/TSV/XLSX) et le convertit en Parquet.
    
    Étapes :
    1. Téléchargement du fichier depuis Supabase Storage
    2. Parsing avec pandas
    3. Validation des colonnes requises
    4. Conversion en Parquet compressé
    5. Upload du Parquet vers Supabase Storage
    6. Calcul des stats échantillon
    7. Mise à jour du statut dans la DB
    """
```

**Retry policy** : 3 tentatives avec backoff exponentiel (1min, 2min, 4min).

### 2. Analyse différentielle (DESeq2)

```python
@app.task(bind=True, max_retries=5, queue="analysis")
def run_deseq_analysis(self, analysis_id: str):
    """
    Exécute DESeq2 via R pour l'analyse d'expression différentielle.
    
    Étapes :
    1. Récupération des datasets et metadata
    2. Préparation du script R (DESeq2 pipeline)
    3. Exécution de Rscript
    4. Parsing des résultats (log2FC, pvalue, padj)
    5. Sauvegarde dans deg_gene table + Parquet
    6. Enregistrement de la provenance (package versions)
    """
```

**Retry policy** : 5 tentatives (les analyses R peuvent échouer pour des raisons mémoire).

### 3. Analyse GSEA

```python
@app.task(bind=True, max_retries=3, queue="analysis")
def run_gsea_analysis(self, analysis_id: str):
    """
    Exécute l'analyse GSEA avec les gene sets annotés.
    
    Étapes :
    1. Récupération de la liste de gènes rangée (ranked)
    2. Chargement des gene sets pertinents
    3. Calcul du NES (Normalized Enrichment Score)
    4. Permutations pour p-value empirique
    5. Correction FDR
    6. Sauvegarde dans gsea_result table + Parquet
    """
```

### 4. Enrichissement GO

```python
@app.task(bind=True, max_retries=3, queue="analysis")
def compute_go_enrichment(self, analysis_id: str):
    """
    Calcule l'enrichissement GO (Fisher exact test) sur les DEG.
    
    Étapes :
    1. Récupération des DEG significatifs
    2. Pour chaque terme GO : table de contingence
    3. Test hypergéométrique / Fisher exact
    4. Correction FDR de Benjamini-Hochberg
    5. Sauvegarde dans enrichment_pathway table + Parquet
    """
```

### 5. Clustering

```python
@app.task(bind=True, max_retries=3, queue="analysis")
def run_clustering(self, dataset_id: str, method: str, n_clusters: int):
    """
    Exécute un clustering (hierarchical ou kmeans) sur les données.
    
    Méthodes : hierarchical, kmeans
    Retourne : labels de cluster + matrice de distance
    """
```

---

## Monitoring des tâches

### Statut d'une tâche

Depuis l'API, on peut vérifier le statut d'une tâche Celery :

```python
from celery.result import AsyncResult

result = AsyncResult(task_id)
print(result.status)  # PENDING, STARTED, RETRY, FAILURE, SUCCESS
print(result.result)  # Résultat ou exception
```

### Logs des workers

```bash
# Logs en temps réel d'un worker
docker-compose logs -f worker

# Ou localement
celery -A app.worker.celery_app worker --loglevel=info
```

### Métriques Celery (Prometheus)

Les métriques sont exposées via le Prometheus endpoint global. Métriques clés :
- `celery_task_started_total` — Nombre de tâches démarrées
- `celery_task_success_total` — Nombre de tâches réussies
- `celery_task_failure_total` — Nombre de tâches échouées
- `celery_task_duration_seconds` — Distribution des durées

---

## Scaling des workers

### En développement (docker-compose.yml)

```yaml
worker:
  build: .
  command: celery -A app.worker.celery_app worker --loglevel=info -Q data_processing,analysis
  depends_on:
    - redis
    - db
```

### En production (docker-compose.prod.yml)

```yaml
worker:
  deploy:
    replicas: 3
  command: >
    celery -A app.worker.celery_app worker --loglevel=info
    -Q data_processing,analysis
    --concurrency=4
```

### Auto-scaling recommandé

| Métrique | Scale up | Scale down |
|---|---|---|
| Queue length (Redis) | > 10 tasks | < 2 tasks |
| Worker CPU usage | > 80% | < 30% |
| Memory usage | > 75% | < 40% |

---

## Gestion des erreurs

### Retry automatique

Toutes les tâches ont `bind=True` pour accéder au contexte de retry :

```python
@app.task(bind=True, max_retries=3)
def my_task(self, ...):
    try:
        do_heavy_work()
    except Exception as exc:
        # Backoff exponentiel : 1min, 2min, 4min
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 60)
```

### Tâches échouées en dernier recours

Les tâches qui ont épuisé leurs retries sont envoyées vers une queue `failed` :

```python
celery_app.conf.task_annotations = {
    '*': {'rate_limit': '10/m'}  # Limite globale pour éviter la surcharge
}
```

### Inspection des workers

```bash
# Lister les workers connectés
celery -A app.worker.celery_app inspect ping

# Voir les tâches en cours
celery -A app.worker.celery_app inspect active

# Voir les tâches en file d'attente
celery -A app.worker.celery_app inspect reserved

# Voir les statistiques des workers
celery -A app.worker.celery_app inspect stats
```

---

## Ordre de dépendance des tâches

Certaines analyses doivent être exécutées dans un ordre précis :

```
1. ingest_dataset(dataset_id)          → Dataset prêt
2. run_deseq_analysis(analysis_id)     → DEG results (dépend de 1)
3a. compute_go_enrichment(analysis_id) → GO enrichment (dépend de 2)
3b. run_gsea_analysis(analysis_id)     → GSEA results (dépend de 2)
```

L'endpoint API orchestre cet enchaînement :

```python
# Après le lancement DESeq2
if should_compute_go:
    compute_go_enrichment.delay(str(analysis_id))
    
if should_compute_gsea:
    run_gsea_analysis.delay(str(analysis_id))
```

---

## Configuration avancée

### Variables d'environnement Celery

| Variable | Description | Défaut |
|---|---|---|
| `CELERY_BROKER_URL` | URL du broker Redis | `REDIS_URL` |
| `CELERY_RESULT_BACKEND` | URL du backend de résultats | `REDIS_URL` |
| `CELERY_TASK_ALWAYS_EAGER` | Mode eager (dev) | `False` |
| `CELERY_WORKER_CONCURRENCY` | Workers par process | `4` |
| `CELERY_TASK_ACKS_LATE` | Ack après exécution | `True` |

### Time limits

```python
celery_app.conf.task_time_limits = (300, 600)  # Soft at 5min, hard at 10min
celery_app.conf.worker_max_tasks_per_child = 100  # Restart worker après 100 tasks (memory leak prevention)
```

---

## Debugging

### Mode eager (exécution synchrone pour le debug)

```python
# .env
CELERY_TASK_ALWAYS_EAGER=True
```

Les tâches sont alors exécutées immédiatement dans le thread appelant.

### Inspection en temps réel

```python
from celery.app.control import Inspect

i = Inspect(app=celery_app)
print(i.active())    # Tâches en cours
print(i.reserved())  # Tâches en file
print.i.scheduled()) # Tâches planifiées ( ETA )
```

### Revocation de tâches

```python
from celery.app.control import Control

c = Control(app=celery_app)
c.revoke(task_id, terminate=True)  # Annule et tue la tâche
c.purge(queue="analysis")          # Vide la file d'attente
```