# GenoLens Next - Guide de Démarrage en Français 🇫🇷

Plateforme SaaS bioinformatique pour l'analyse de données transcriptomiques.

## 🎯 Vue d'Ensemble

GenoLens Next est une plateforme backend **"Bring Your Own Data"** qui permet aux chercheurs d'uploader leurs données transcriptomiques déjà traitées (matrices de comptage, résultats de différentiel d'expression, enrichissements) et de les interroger à la demande.

### Caractéristiques Principales

- ✅ **Architecture Asset-Based**: Métadonnées en PostgreSQL, données en Parquet
- ✅ **Chargement Lazy**: Données chargées à la demande avec filtres
- ✅ **Processing en Background**: Conversion asynchrone CSV → Parquet
- ✅ **Supabase Integration**: Authentification et stockage S3-compatible
- ✅ **Haute Performance**: FastAPI avec SQLAlchemy async

## 🚀 Démarrage Rapide (5 minutes)

### Prérequis
- Docker & Docker Compose installés
- Compte Supabase (gratuit)

### Étape 1: Configuration Supabase

Copiez `.env.example` vers `.env` et renseignez vos informations Supabase:
- `SUPABASE_URL` — URL de votre projet Supabase
- `SUPABASE_KEY` — Clé anon publique
- `SUPABASE_JWT_SECRET` — Secret JWT (onglet API Settings)

### Étape 2: Créer le Bucket Storage (2 minutes)

1. Allez sur: https://supabase.com/dashboard/project/<votre-project-ref>/storage/buckets
2. Cliquez sur **"New bucket"**
3. Nom: `genolens-data`
4. Type: **Private** ✓
5. Cliquez sur **"Create bucket"**

### Étape 3: Démarrer les Services (3 minutes)

```bash
# Lancer la configuration automatique
make setup

# Ou manuellement:
docker-compose up -d
docker-compose exec api alembic upgrade head
```

### Étape 4: Vérifier

```bash
# Vérifier que tout fonctionne
make verify

# Ou tester manuellement:
curl http://localhost:8000/health
```

## 🌐 Services Disponibles

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | Documentation interactive OpenAPI |
| **Health** | http://localhost:8000/health | Statut de l'API |
| **Flower** | http://localhost:5555 | Monitoring des tâches Celery |

## 📚 Documentation

### Guides de Démarrage
- **[START_HERE.md](START_HERE.md)** ⭐ **COMMENCER ICI**
- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Instructions détaillées
- **[SUPABASE_CONFIG.md](SUPABASE_CONFIG.md)** - Configuration Supabase

### Documentation Technique
- **[README.md](README.md)** - Documentation complète (EN)
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Architecture détaillée
- **[QUICKSTART.md](QUICKSTART.md)** - Guide rapide

## 🛠️ Commandes Principales

### Démarrage
```bash
make setup      # Configuration complète
make up         # Démarrer les services
make down       # Arrêter les services
make restart    # Redémarrer
```

### Développement
```bash
make logs       # Voir tous les logs
make logs-api   # Logs de l'API uniquement
make shell      # Accéder au container
make test-api   # Tester l'API
```

### Base de Données
```bash
make migrate    # Appliquer les migrations
make migration  # Créer une nouvelle migration
make db-shell   # Shell PostgreSQL
```

### Aide
```bash
make help       # Voir toutes les commandes
```

## 🧪 Tester l'API

### 1. Créer un Utilisateur

**Via l'interface Supabase:**
1. https://supabase.com/dashboard/project/<votre-project-ref>/auth/users
2. Cliquez sur "Add user" → "Create new user"
3. Email: `test@genolens.com`
4. Password: `TestPassword123!`

**Via l'API:**
```bash
curl -X POST 'https://<votre-project-ref>.supabase.co/auth/v1/signup' \
  -H "apikey: <votre-anon-key>" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@genolens.com","password":"TestPassword123!"}'
```

### 2. Obtenir un Token

```bash
curl -X POST 'https://<votre-project-ref>.supabase.co/auth/v1/token?grant_type=password' \
  -H "apikey: <votre-anon-key>" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@genolens.com","password":"TestPassword123!"}'
```

### 3. Créer un Projet

```bash
curl -X POST "http://localhost:8000/api/v1/projects/" \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Mon Projet","description":"Test GenoLens"}'
```

### 4. Uploader un Dataset

```bash
# Créer un fichier test
cat > test_counts.csv << EOF
gene_id,Sample_A,Sample_B,Sample_C
ENSG00000001,100,150,120
ENSG00000002,200,180,210
EOF

# Uploader
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer VOTRE_TOKEN" \
  -F "project_id=VOTRE_PROJECT_ID" \
  -F "name=Matrice de Comptage" \
  -F "dataset_type=MATRIX" \
  -F "file=@test_counts.csv"
```

### 5. Interroger les Données

```bash
curl "http://localhost:8000/api/v1/datasets/DATASET_ID/query?limit=10" \
  -H "Authorization: Bearer VOTRE_TOKEN"
```

## 🏗️ Architecture

### Modèle Asset-Based

```
┌─────────────────┐
│  PostgreSQL     │  → Métadonnées uniquement
│  (Léger)        │     • Projects
└─────────────────┘     • Samples
                        • Dataset refs
                        • File paths

┌─────────────────┐
│ Supabase Storage│  → Données réelles
│ (Parquet)       │     • Raw CSV/Excel
└─────────────────┘     • Processed Parquet
```

**Avantages:**
- ✅ Scalable à des milliards de points
- ✅ Base de données légère
- ✅ Requêtes ultra-rapides
- ✅ Schéma flexible par dataset

### Flux de Données

**Upload:**
```
User → FastAPI → Supabase Storage (CSV) → Celery Worker
     → Conversion Parquet → Upload Parquet → DB Update (READY)
```

**Query:**
```
User → FastAPI → Download Parquet → Pandas (filtres)
     → JSON Response
```

## 🎯 Endpoints API

### Projets
- `POST /api/v1/projects/` - Créer un projet
- `GET /api/v1/projects/` - Lister les projets (paginé)
- `GET /api/v1/projects/{id}` - Obtenir un projet
- `PATCH /api/v1/projects/{id}` - Mettre à jour
- `DELETE /api/v1/projects/{id}` - Supprimer

### Datasets
- `POST /api/v1/datasets/upload` - Uploader CSV/Excel/TSV
- `GET /api/v1/datasets/{id}` - Obtenir les métadonnées
- `GET /api/v1/datasets/{id}/query` - Interroger avec filtres
- `GET /api/v1/datasets/project/{id}` - Lister les datasets

## 🔧 Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Framework | FastAPI (Python 3.11+) |
| Base de données | PostgreSQL 15 |
| ORM | SQLAlchemy 2.0 (Async) |
| Migrations | Alembic |
| Auth | Supabase Auth (JWT) |
| Queue | Celery + Redis |
| Data | Pandas + PyArrow |
| Storage | Supabase Storage |
| Container | Docker Compose |

## 📁 Structure du Projet

```
genolens_v2/
├── app/
│   ├── api/endpoints/     # Routes API
│   ├── core/              # Configuration
│   ├── db/                # Database
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic
│   ├── worker/            # Celery tasks
│   └── main.py            # FastAPI app
├── alembic/               # Migrations
├── scripts/               # Scripts utiles
├── docker-compose.yml     # Services
└── .env                   # Configuration ✓
```

## 🐛 Dépannage

### L'API ne démarre pas
```bash
docker-compose logs api
# Attendre 30 secondes
docker-compose restart api
```

### Le Worker ne traite pas
```bash
docker-compose logs worker
docker-compose restart worker
```

### Erreurs Supabase Storage
- Vérifier que le bucket existe
- Vérifier qu'il est "Private"
- Vérifier les clés dans `.env`

## 📞 Support

- **Documentation**: Voir [START_HERE.md](START_HERE.md)
- **Issues**: GitHub Issues
- **Supabase Dashboard**: https://supabase.com/dashboard/project/<votre-project-ref>

## 🎓 Ressources

### Scripts Automatiques
```bash
./scripts/quick_setup.sh   # Setup complet
./scripts/verify_setup.sh  # Vérification
./scripts/test_api.sh      # Tests API
```

### Liens Rapides
- **Dashboard Supabase**: https://supabase.com/dashboard/project/<votre-project-ref>
- **Storage**: https://supabase.com/dashboard/project/<votre-project-ref>/storage/buckets
- **Auth**: https://supabase.com/dashboard/project/<votre-project-ref>/auth/users

## ✅ Checklist de Démarrage

- [x] Configuration Supabase dans `.env`
- [ ] Créer le bucket `genolens-data`
- [ ] Lancer `make setup`
- [ ] Vérifier avec `make verify`
- [ ] Créer un utilisateur de test
- [ ] Tester avec `make test-api`

## 🚀 Prochaines Étapes

1. **Créer le bucket Storage** (2 min)
2. **Démarrer avec `make setup`** (3 min)
3. **Tester l'API** → http://localhost:8000/docs
4. **Développer le frontend** → Connecter votre app React/Next.js
5. **Déployer en production** → Suivre le guide de déploiement

---

**Tout est prêt! Il ne reste plus qu'à créer le bucket et lancer `make setup`!** 🎉

📖 **Lisez [START_HERE.md](START_HERE.md) pour commencer!**
