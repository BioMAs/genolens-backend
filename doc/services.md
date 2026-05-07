# Services Métier - GenoLens Backend

## Vue d'ensemble

Les services sont situés dans `app/services/` et encapsulent toute la logique métier. Chaque service est un module Python indépendant, généralement avec des méthodes statiques ou un singleton.

---

## data_processor.py

**Rôle** : Parsing et conversion de fichiers biologiques (CSV, TSV, XLSX) en format Parquet optimisé.

### Fonctions principales

```python
class DataProcessor:
    @staticmethod
    async def parse_and_validate(file_path, file_type, metadata):
        """Parse un fichier uploadé et retourne les métadonnées."""
        
    @staticmethod
    async def convert_to_parquet(df, dataset_id, compression="snappy"):
        """Convertit un DataFrame pandas en fichier Parquet compressé."""
        
    @staticmethod
    async def read_parquet_chunked(dataset_id, chunk_size=10000):
        """Lit un fichier Parquet par chunks pour le traitement mémoire efficace."""
        
    @staticmethod
    async def compute_sample_stats(df):
        """Calcule les statistiques des échantillons (mean, median, std, etc.)."""
```

### Formats supportés

| Format | Extension | Parser |
|---|---|---|
| CSV | `.csv` | pandas.read_csv |
| TSV | `.tsv`, `.txt` | pandas.read_csv(sep='\t') |
| Excel | `.xlsx` | openpyxl / pandas |
| Parquet (output) | `.parquet` | pyarrow.parquet |

### Validation des données

- Vérification des colonnes requises (gene_id, counts, metadata samples)
- Détection automatique du séparateur pour les fichiers texte
- Gestion des valeurs manquantes et incohérentes
- Normalisation des noms de gènes/symboles

---

## analysis_service.py

**Rôle** : Orchestration des analyses bioinformatiques (DEG, clustering, etc.).

### Fonctions principales

```python
class AnalysisService:
    @staticmethod
    async def create_analysis_run(project_id, user_id, analysis_type, parameters):
        """Crée un enregistrement d'analyse avec provenance."""
        
    @staticmethod
    async def get_results(analysis_id):
        """Récupère les résultats d'une analyse (fichier Parquet ou JSON)."""
        
    @staticmethod
    async def get_analysis_history(project_id, user_id=None):
        """Liste toutes les analyses d'un projet."""
```

### Provenance des analyses

Chaque `AnalysisRun` enregistre :
- Les paramètres exacts utilisés
- Les versions des packages R/Python
- Le chemin vers le fichier résultat
- Le statut (pending → running → completed/failed)

---

## go_service.py

**Rôle** : Gestion de l'ontologie GO (Gene Ontology) et calcul d'enrichissement.

### Fonctions principales

```python
class GOService:
    @staticmethod
    async def get_term(term_id):
        """Récupère un terme GO par son ID (GO:XXXXXXX)."""
        
    @staticmethod
    async def get_children(term_id):
        """Récupère les termes enfants d'un terme GO."""
        
    @staticmethod
    async def get_ancestors(term_id):
        """Récupère les ancêtres d'un terme GO (pour le DAG)."""
        
    @staticmethod
    async def search_terms(query, ontology="BP|MF|CC"):
        """Recherche de termes GO par nom ou ID."""
```

### Structure du DAG GO

```
GO:0008150 (biological_process)
├── GO:0006915 (apoptotic process)
│   ├── GO:0006917 (induction of apoptosis)
│   └── GO:0043067 (regulation of programmed cell death)
├── GO:0007049 (cell cycle)
    └── ...
```

---

## go_loader.py / gene_set_loader.py

**Rôle** : Chargement et synchronisation des données GO et gene sets depuis les sources externes.

### Sources de données

| Source | Contenu | Fréquence sync |
|---|---|---|
| GO (Gene Ontology) | Termes, relations parent-enfant | Mensuelle |
| GOA (GO Annotations) | Annotations gène → terme GO | Mensuelle |
| MSigDB | Gene sets pour GSEA | Versionnée |

### Scripts de chargement

```bash
# Charger l'ontologie GO complète
./scripts/load_go_ontology.py

# Charger les annotations GOA
./scripts/load_go_annotations_from_goa.py

# Charger les gene sets pour GSEA
./scripts/load_gene_sets.py

# Synchroniser les tables GO
./scripts/sync_go_tables.py
```

---

## gsea_processor.py

**Rôle** : Traitement et calcul des analyses GSEA (Gene Set Enrichment Analysis).

### Fonctions principales

```python
class GSEAProcessor:
    @staticmethod
    async def run_gsea(ranked_gene_list, gene_sets, n_permutations=1000):
        """Exécute le calcul GSEA et retourne les résultats."""
        
    @staticmethod
    async def compute_nes(ranking, gene_set):
        """Calcule le Normalized Enrichment Score (NES)."""
        
    @staticmethod
    async def compute_fdr(results):
        """Calcule le FDR q-value à partir des permutations."""
```

### Résultats GSEA

| Champ | Description |
|---|---|
| `term_id` | ID du gene set / pathway |
| `term_name` | Nom du pathway |
| `nes` | Normalized Enrichment Score |
| `p_value` | p-value empirique |
| `fdr_qvalue` | FDR corrigé |
| `leading_edge` | Gènes contributeurs principaux |
| `regulation` | "up" ou "down" (direction de l'enrichissement) |

---

## clustering_service.py

**Rôle** : Algorithmes de clustering pour les données transcriptomiques.

### Méthodes supportées

- **Hierarchical Clustering** : avec différents liens (average, complete, ward)
- **KMeans Clustering** : avec optimisation du nombre de clusters (silhouette score)
- **DBSCAN** : pour la détection de clusters de densité variable

```python
class ClusteringService:
    @staticmethod
    async def hierarchical_clustering(data_matrix, method="average", n_clusters=None):
        """Clusterage hiérarchique avec dendrogramme."""
        
    @staticmethod
    async def kmeans_clustering(data_matrix, n_clusters=5, max_iter=300):
        """Clusterage KMeans avec calcul du silhouette score."""
```

---

## stats_service.py

**Rôle** : Calculs statistiques pour les analyses bioinformatiques.

### Fonctions principales

```python
class StatsService:
    @staticmethod
    def compute_deg_stats(deg_results):
        """Calcule les stats DEG (log2FC, p-value adj, etc.)."""
        
    @staticmethod
    def benjamini_hochberg(p_values):
        """Correction FDR de Benjamini-Hochberg."""
        
    @staticmethod
    def fisher_exact_test(contingency_table):
        """Test d'enrichissement hypergéométrique/Fisher exact."""
```

---

## cache_service.py / persistent_cache_service.py

**Rôle** : Caching distribué Redis pour les computations coûteuses.

### Cache TTL (transitoire)

```python
# Initialisation
await cache_service.initialize(redis_url="redis://localhost:6379/0")

# Utilisation avec décorateur
@cache_service.ttl_cache(ttl=3600, key_prefix="clustering")
async def get_clustering(dataset_id):
    return compute_clustering(dataset_id)  # Seulement si pas en cache
```

### Caches implémentés

| Cache | TTL | Clé préfixe | Usage |
|---|---|---|---|
| Clustering | 1h | `clustering` | Résultats de clustering par dataset |
| Volcano | 2h | `volcano` | Stats volcano plot |
| Sample stats | 4h | `sample_stats` | Stats des échantillons |
| GSEA | 6h | `gsea` | Résultats GSEA |

### Cache persistant

```python
class PersistentCacheService:
    async def get_or_compute(key, compute_fn, ttl=3600):
        """Récupère du cache ou calcule et stocke."""
        
    async def invalidate(key):
        """Invalide une entrée du cache."""
```

---

## ai_interpreter.py

**Rôle** : Interface avec Ollama pour l'interprétation biologique des résultats.

### Fonctions principales

```python
class AIInterpreter:
    @staticmethod
    async def interpret_results(analysis_data, context):
        """Génère une interprétation biologique des résultats."""
        
    @staticmethod
    async def suggest_visualizations(analysis_type, results_summary):
        """Suggère les visualisations pertinentes."""
        
    @staticmethod
    async def generate_code(analysis_request, language="R"):
        """Génère du code R/Python pour une analyse personnalisée."""
```

### Prompts système

Chaque type d'interprétation utilise un prompt spécifique :
- **DEG interpretation** : Focus sur les gènes top, pathways enrichis
- **Enrichment interpretation** : Synthèse des voies biologiques significatives
- **Visualization suggestion** : Recommandation basée sur le type de données et l'analyse

---

## stripe_service.py

**Rôle** : Intégration avec Stripe pour la facturation et les abonnements.

### Fonctions principales

```python
class StripeService:
    @staticmethod
    async def create_checkout_session(user_id, price_id, success_url, cancel_url):
        """Crée une session de checkout Stripe."""
        
    @staticmethod
    async def get_subscription(user_id):
        """Récupère l'abonnement actuel d'un utilisateur."""
        
    @staticmethod
    async def handle_webhook(payload, signature):
        """Traite un webhook Stripe (validation + action)."""
        
    @staticmethod
    async def create_customer_portal_session(customer_id, return_url):
        """Crée une session de gestion du compte client Stripe."""
```

### Plans supportés

| Plan | Price ID (config) | Features |
|---|---|---|
| Free | — | Quota limité, analyses basiques |
| PREMIUM | `stripe_price_premium_monthly` | Analyses avancées, plus de datasets |
| ADVANCED | `stripe_price_advanced_monthly` | Tout illimité, support prioritaire |

---

## bookmarks_service.py

**Rôle** : Gestion des bookmarks (sauvegardes de sélections).

```python
class BookmarkService:
    @staticmethod
    async def create(user_id, dataset_id, name, gene_list=None):
        """Crée un bookmark avec option de liste de gènes."""
        
    @staticmethod
    async def get_all(user_id):
        """Liste tous les bookmarks d'un utilisateur."""
        
    @staticmethod
    async def delete(bookmark_id, user_id):
        """Supprime un bookmark (seul l'owner peut le faire)."""
```

---

## comments_service.py

**Rôle** : Gestion des commentaires sur les projets.

```python
class CommentService:
    @staticmethod
    async def create(project_id, user_id, content):
        """Ajoute un commentaire à un projet."""
        
    @staticmethod
    async def get_by_project(project_id, user_id):
        """Récupère tous les commentaires d'un projet (avec accès vérifié)."""
        
    @staticmethod
    async def update(comment_id, user_id, content):
        """Modifie un commentaire (seul l'owner peut le faire)."""
```

---

## history_service.py

**Rôle** : Historique des actions utilisateur et activité des projets.

```python
class HistoryService:
    @staticmethod
    async def log_action(user_id, action_type, project_id=None, metadata=None):
        """Enregistre une action dans l'historique."""
        
    @staticmethod
    async def get_user_history(user_id, limit=50):
        """Récupère l'historique d'un utilisateur."""
        
    @staticmethod
    async def get_project_activity(project_id):
        """Récupère le journal d'activité d'un projet."""
```

---

## email_service.py

**Rôle** : Envoi d'emails transactionnels via SMTP.

```python
class EmailService:
    @staticmethod
    async def send_welcome_email(user_email, user_name):
        """Envoie un email de bienvenue."""
        
    @staticmethod
    async def send_subscription_confirmation(user_email, plan_details):
        """Confirme un changement d'abonnement."""
```

---

## storage.py

**Rôle** : Abstraction du stockage (Supabase Storage + local filesystem).

```python
class StorageService:
    @staticmethod
    async def upload_file(bucket, path, file_data):
        """Upload un fichier dans le bucket Supabase."""
        
    @staticmethod
    async def download_file(bucket, path):
        """Télécharge un fichier depuis le bucket."""
        
    @staticmethod
    async def delete_file(bucket, path):
        """Supprime un fichier du bucket."""
```

---

## external_integrations.py

**Rôle** : Connexion aux bases de données biologiques externes.

### Intégrations supportées

| Source | Endpoint utilisé | Données récupérées |
|---|---|---|
| UniProt | `api.uniprot.org` | Fonction, localisation, voies d'un gène |
| NCBI | `eutils.ncbi.nlm.nih.gov` | Annotation, synonymes d'un gène |

```python
class ExternalIntegrations:
    @staticmethod
    async def fetch_uniprot(gene_symbol):
        """Récupère les données UniProt pour un gène."""
        
    @staticmethod
    async def fetch_ncbi(gene_symbol, species="human"):
        """Récupère les données NCBI pour un gène."""
```

---

## anno_db_service.py

**Rôle** : Service d'annotation de gènes (mapping entre identifiants).

```python
class AnnoDBService:
    @staticmethod
    async def resolve_gene_id(symbol, species):
        """Résout le symbole d'un gène en ID officiel."""
        
    @staticmethod
    async def get_aliases(gene_id):
        """Retourne tous les alias/synonymes d'un gène."""
```

---

## version_service.py

**Rôle** : Gestion des versions de datasets et traçabilité des modifications.

```python
class VersionService:
    @staticmethod
    async def create_version(dataset_id, change_description):
        """Crée une nouvelle version d'un dataset."""
        
    @staticmethod
    async def get_versions(dataset_id):
        """Liste toutes les versions d'un dataset."""
```

---

## demo_seed_service.py

**Rôle** : Seed de données de démonstration pour le développement et les démos.

```python
class DemoSeedService:
    @staticmethod
    async def seed_demo_data():
        """Crée des projets, datasets et analyses de démonstration."""
        
    @staticmethod
    async def patch_demo_matrix(project_id):
        """Patch un dataset demo avec des données réalistes."""
```

---

## Architecture des services

### Dépendances entre services

```
api/endpoints/
  └──► analysis_service
       ├──► gsea_processor
       ├──► clustering_service
       ├──► stats_service
       ├──► go_service
       │    └──► gene_set_loader
       └──► ai_interpreter
            └──► stripe_service (quota check)

api/endpoints/
  └──► datasets
       ├──► data_processor
       ├──► storage
       └──► version_service
```

### Pattern d'injection

Les services sont utilisés directement via leurs méthodes statiques ou instanciés dans les endpoints :

```python
# Dans un endpoint FastAPI
@router.post("/analyses/differential-expression")
async def run_deg_analysis(
    request: DifferentialExpressionRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    # Vérification quota
    quota_ok = await stripe_service.check_ai_quota(user.id)
    
    # Création de l'analyse
    analysis_id = await analysis_service.create_analysis_run(...)
    
    # Lancement en tâche Celery
    run_deseq_analysis.delay(str(analysis_id))
    
    return {"analysis_id": str(analysis_id), "status": "pending"}
```