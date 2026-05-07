# Base de données - GenoLens Backend

## Vue d'ensemble

La base de données principale est **PostgreSQL** gérée via **SQLAlchemy (async)** et **Alembic** pour les migrations.

### Configuration

```python
# app/core/config.py
DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/genolens"
```

Le driver `asyncpg` est utilisé pour les performances async.

---

## Modèles principaux

### Utilisateurs & Auth

Les utilisateurs sont gérés par **Supabase Auth**. Les données relationnelles complémentaires sont stockées dans les tables suivantes :

| Table | Description |
|---|---|
| `user_subscription` | Plan Stripe, statut, dates de cycle |
| `ai_quota` | Quota d'usage IA par utilisateur |
| `login_events` | Historique des connexions (audit) |

### Projets & Datasets

| Table | Description | Clés étrangères |
|---|---|---|
| `projects` | Projets transcriptomiques | `owner_id → auth.users` |
| `project_members` | Membres partagés d'un projet | `project_id, user_id` |
| `datasets` | Datasets uploadés (metadata) | `project_id → projects` |
| `dataset_metadata` | Metadata JSON des datasets | `dataset_id → datasets` |

**Champs clés de `projects`** :
```python
class Project(Base):
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String(255))
    description = Column(Text)
    owner_id = Column(UUID, ForeignKey("auth.users.id"))
    species = Column(String(100))  # homo_sapiens, mus_musculus...
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
```

### Analyses

| Table | Description | Clés étrangères |
|---|---|---|
| `analysis_run` | Chaque exécution d'analyse | `project_id → projects`, `user_id → auth.users` |
| `self_service_analyses` | Analyses via le wizard | `project_id → projects` |

**Champs clés de `analysis_run`** :
```python
class AnalysisRun(Base):
    id = Column(UUID, primary_key=True)
    project_id = Column(UUID, ForeignKey("projects.id"))
    user_id = Column(UUID, ForeignKey("auth.users.id"))
    analysis_type = Column(String(50))  # deg, gsea, clustering...
    parameters = Column(JSON)  # Paramètres de l'analyse
    result_path = Column(String(500))  # Chemin vers le fichier Parquet résultat
    status = Column(String(20))  # pending, running, completed, failed
    package_versions = Column(JSON)  # Versions R/Python utilisées (provenance)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
```

### Résultats bioinformatiques

| Table | Description |
|---|---|
| `deg_gene` | Gènes différentiellement exprimés |
| `enrichment_pathway` | Résultats d'enrichissement de voies |
| `gsea_result` | Résultats GSEA (NES, p-value...) |
| `go_term` | Termes GO annotés |
| `go_annotation` | Annotations gène → terme GO |

### IA & Conversation

| Table | Description |
|---|---|
| `ai_conversation_history` | Conversations utilisateur-IA |
| `ai_interpretation` | Interprétations automatiques de résultats |

### Bookmarks & Comments

| Table | Description | Clés étrangères |
|---|---|---|
| `bookmark` | Sauvegardes utilisateur | `user_id → auth.users`, `dataset_id → datasets` |
| `project_comment` | Commentaires sur les projets | `project_id → projects`, `user_id → auth.users` |

### Activity & Provenance

| Table | Description |
|---|---|
| `project_activity_log` | Journal d'activité des projets |
| `cached_computation` | Cache persistant des computations coûteuses |

---

## Schéma de base de données

Le schéma complet est disponible dans :
- **`sql/supabase_complete_schema.sql`** — Schéma complet de référence
- **`sql/supabase_deg_genes_schema.sql`** — Schéma DEG/Gènes différentiels
- **`sql/supabase_enrichment_pathways_schema.sql`** — Schéma enrichissement voies
- **`sql/supabase_schema_safe.sql`** — Schéma sécurisé (RLS policies)

---

## Migrations Alembic

### Configuration

```ini
# alembic.ini
script_location = alembic
sqlalchemy.url = postgresql+asyncpg://postgres:postgres@localhost:5432/genolens
```

### Commandes courantes

```bash
# Créer une nouvelle migration
alembic revision --autogenerate -m "description de la migration"

# Appliquer les migrations
alembic upgrade head

# Revenir en arrière d'une version
alembic downgrade -1

# Voir le statut actuel
alembic current

# Voir l'historique des versions
alembic history --verbose
```

### Dernières migrations notables

| Date | Migration | Description |
|---|---|---|
| 2026-04-22 | `add_self_service_analyses_table` | Table analyses wizard auto-service |
| 2026-04-21 | `add_analysis_runs_table` | Provenance des analyses |
| 2026-04-13 | `user_login_events` | Audit login utilisateurs |
| 2026-02-26 | `add_go_terms_and_annotations_tables` | Tables GO ontology |
| 2026-02-25 | `performance_optimizations` | Index et optimisations de performance |
| 2026-01-08 | `add_regulation_to_enrichment` | Champ regulation dans enrichment |
| 2026-01-03 | `add_deggene_and_enrichmentpathway_models` | Modèles DEG et enrichissement |

---

## Index & Performance

Les index suivants sont créés automatiquement via les migrations de performance :

```sql
-- Projets
CREATE INDEX idx_projects_owner_id ON projects(owner_id);
CREATE INDEX idx_projects_created_at ON projects(created_at DESC);

-- Datasets
CREATE INDEX idx_datasets_project_id ON datasets(project_id);

-- Analyses
CREATE INDEX idx_analysis_run_project_id ON analysis_run(project_id);
CREATE INDEX idx_analysis_run_user_id ON analysis_run(user_id);
CREATE INDEX idx_analysis_run_status ON analysis_run(status);

-- DEG genes
CREATE INDEX idx_deg_gene_analysis_id ON deg_gene(analysis_id);

-- Enrichment
CREATE INDEX idx_enrichment_pathway_analysis_id ON enrichment_pathway(analysis_id);

-- Bookmarks
CREATE INDEX idx_bookmark_user_id ON bookmark(user_id);

-- Comments
CREATE INDEX idx_project_comment_project_id ON project_comment(project_id);
```

---

## Supabase RLS Policies

Les politiques Row-Level Security (RLS) sont configurées pour :

- **Projets** : seuls l'owner et les membres peuvent accéder
- **Datasets** : accès restreint aux membres du projet parent
- **Bookmarks** : chaque utilisateur ne voit que ses bookmarks
- **Comments** : lecture publique au sein du projet, écriture owner/admin
- **AI conversations** : isolation complète par user_id

---

## Connexion en production

```python
# docker-compose.prod.yml utilise :
DATABASE_URL: postgresql+asyncpg://genolens:${DB_PASSWORD}@db:5432/genolens_prod
```

Recommandations pour la production :
- Utiliser **PgBouncer** en connection pooling
- Configurer le `max_overflow` et `pool_size` dans SQLAlchemy
- Activer les backups automatisés PostgreSQL
- Utiliser un SSL/TLS pour la connexion DB

---

## Backup & Restore

Un script de backup est fourni :

```bash
# Lancer le backup
./backup.sh

# Le script effectue :
# 1. pg_dump de la base genolens
# 2. Compression gzip
# 3. Rotation des backups (garde les 7 derniers)
```

---

## Migration depuis Supabase

Le schéma initial peut être importé depuis une instance Supabase existante :

```bash
./scripts/import_schema_from_supabase.sh
```