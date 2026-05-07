# Documentation Backend - GenoLens

Cette section contient la documentation technique détaillée du backend de la plateforme GenoLens.

## 📑 Table des matières

- [Architecture](architecture.md)
- [API Endpoints](api-endpoints.md)
- [Base de données](database.md)
- [Services métier](services.md)
- [Worker Celery](celery-worker.md)
- [Sécurité](security.md)
- [Déploiement](deployment.md)

## 🚀 Vue d'ensemble

Le backend est une API RESTful construite avec **FastAPI** (Python 3.10+). Il gère :

- L'authentification via Supabase Auth
- Le stockage hybride (PostgreSQL pour les métadonnées, fichiers Parquet pour les données biologiques)
- Le traitement asynchrone via Celery + Redis
- L'intégration IA avec Ollama
- La gestion des abonnements Stripe
- Le monitoring (Sentry, Prometheus, Grafana)

## 📂 Structure du projet

```
backend/
├── alembic/              # Migrations de base de données
├── app/
│   ├── api/
│   │   ├── deps/         # Dépendances FastAPI (auth, db, etc.)
│   │   └── endpoints/    # Routes API (projects, datasets, analyses...)
│   ├── core/             # Config, sécurité, monitoring
│   ├── db/               # Session SQLAlchemy
│   ├── middleware/       # Middlewares (CORS, rate limiting, security headers)
│   ├── models/           # Modèles SQLAlchemy
│   ├── schemas/          # Schémas Pydantic (validation request/response)
│   ├── services/         # Logique métier
│   └── worker/           # Tasks Celery
├── scripts/              # Scripts utilitaires
├── sql/                  # Schémas SQL de référence
├── tests/                # Tests pytest
├── monitoring/           # Configuration Prometheus/Grafana
├── docker-compose.yml    # Environnement de développement
├── docker-compose.prod.yml  # Déploiement production
└── pyproject.toml        # Configuration Python (mypy, black, isort, pytest)
```

## 🔧 Technologies principales

| Technologie | Usage |
|---|---|
| FastAPI | Framework API RESTful |
| SQLAlchemy (async) | ORM base de données |
| Alembic | Migrations de schéma |
| Celery + Redis | Tâches asynchrones |
| PostgreSQL | Base de données principale |
| Pydantic | Validation des données |
| Pandas + PyArrow | Traitement de données biologiques |
| Supabase Auth | Authentification utilisateurs |
| Stripe | Paiements et abonnements |
| Ollama | Interprétation IA locale |
| Sentry | Monitoring d'erreurs |
| Prometheus | Métriques de performance |

## 📖 Documentation détaillée

Consultez les fichiers dans ce dossier pour des détails approfondis :

- **[Architecture](architecture.md)** - Vue d'ensemble de l'architecture, flux de données, patterns utilisés
- **[API Endpoints](api-endpoints.md)** - Référence complète des endpoints REST
- **[Base de données](database.md)** - Schéma, modèles, migrations
- **[Services métier](services.md)** - Description des services (data processing, AI, enrichment...)
- **[Worker Celery](celery-worker.md)** - Configuration et tâches asynchrones
- **[Sécurité](security.md)** - Auth, rate limiting, CORS, headers de sécurité
- **[Déploiement](deployment.md)** - Docker, production, CI/CD

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Avec couverture
pytest --cov=app --cov-report=html

# Tests unitaires uniquement
pytest -m unit

# Tests d'intégration
pytest -m integration
```

## 🔗 Liens utiles

- [Swagger UI (docs)](http://localhost:8000/docs) — Documentation interactive de l'API
- [ReDoc](http://localhost:8000/redoc) — Documentation alternative
- [Prometheus Metrics](http://localhost:8000/metrics) — Métriques de performance