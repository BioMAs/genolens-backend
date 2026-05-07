# API Endpoints - GenoLens Backend

## Vue d'ensemble

L'API expose un prefix `/api/v1` avec des endpoints RESTful pour toutes les fonctionnalités de la plateforme.

---

## Health & System

### `GET /`
**Tag**: Health  
Retourne les informations de l'application.

```json
{
  "app": "GenoLens Next",
  "version": "1.0.0",
  "environment": "development",
  "status": "running",
  "docs": "/docs"
}
```

### `GET /health`
**Tag**: Health  
Health check pour le monitoring.

```json
{ "status": "healthy", "app": "GenoLens Next", "version": "1.0.0" }
```

### `GET /db-test`
**Tag**: Health (dev only)  
Test de connectivité base de données. Retourne les counts de projets/datasets.

---

## Projects (`/api/v1/projects`)

Gestion des projets transcriptomiques.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/projects` | Liste des projets (paginée) | ✅ |
| `POST` | `/projects` | Créer un nouveau projet | ✅ |
| `GET` | `/projects/{id}` | Détails d'un projet | ✅ |
| `PATCH` | `/projects/{id}` | Mettre à jour un projet | ✅ (owner) |
| `DELETE` | `/projects/{id}` | Supprimer un projet | ✅ (owner) |
| `GET` | `/projects/{id}/stats` | Statistiques du projet | ✅ |
| `POST` | `/projects/{id}/members` | Ajouter un membre | ✅ (owner/admin) |
| `DELETE` | `/projects/{id}/members/{user_id}` | Retirer un membre | ✅ (owner/admin) |

**Schéma ProjectCreate** :
```json
{
  "name": "My Analysis",
  "description": "Differential expression analysis",
  "species": "homo_sapiens"
}
```

---

## Datasets (`/api/v1/datasets`)

Gestion des datasets (upload, metadata, processing).

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/datasets/upload` | Upload un fichier CSV/TSV/XLSX | ✅ |
| `GET` | `/datasets/{id}` | Metadata du dataset | ✅ (project access) |
| `PATCH` | `/datasets/{id}` | Mettre à jour les metadata | ✅ |
| `DELETE` | `/datasets/{id}` | Supprimer un dataset | ✅ |
| `POST` | `/datasets/{id}/query` | Query des données avec filtres | ✅ |
| `GET` | `/projects/{project_id}/datasets` | Lister datasets d'un projet | ✅ |

**Upload** : `multipart/form-data`, max 500 MB. Retourne un `dataset_id` et le statut de processing.

---

## Analyses (`/api/v1/analyses`)

Lancement et suivi des analyses bioinformatiques.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/analyses/differential-expression` | Lancer DESeq2 | ✅ |
| `GET` | `/analyses/{id}` | Statut et résultats d'une analyse | ✅ |
| `GET` | `/analyses/project/{project_id}` | Liste des analyses d'un projet | ✅ |
| `POST` | `/analyses/self-service` | Analyse auto-service (wizard) | ✅ |

**Schéma DifferentialExpressionRequest** :
```json
{
  "dataset_ids": ["uuid-1", "uuid-2"],
  "comparison": {
    "group1": "control",
    "group2": "treated"
  },
  "species": "homo_sapiens",
  "multiple_testing_correction": "fdr"
}
```

---

## Enrichment (`/api/v1/enrichment`)

Analyses d'enrichissement GO et KEGG.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/enrichment/go` | Enrichissement GO (Fisher test) | ✅ |
| `GET` | `/enrichment/{analysis_id}` | Résultats enrichissement | ✅ |
| `POST` | `/enrichment/gsea` | Analyse GSEA | ✅ |
| `GET` | `/enrichment/gsea/{analysis_id}` | Résultats GSEA | ✅ |

**Schéma GOEnrichmentRequest** :
```json
{
  "gene_list": ["TP53", "BRCA1", "EGFR"],
  "ontology": "BP",
  "species": "homo_sapiens",
  "pvalue_threshold": 0.05,
  "min_gene_set_size": 10,
  "max_gene_set_size": 500
}
```

---

## Genes (`/api/v1/genes`)

Recherche et annotation de gènes.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/genes/search` | Recherche de gènes (nom, symbole) | ✅ |
| `GET` | `/genes/{gene_id}` | Détails d'un gène | ✅ |
| `GET` | `/genes/symbols/validate` | Valider une liste de symboles | ✅ |

---

## Ontology (`/api/v1/ontology`)

Navigation dans l'ontologie GO.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/ontology/terms` | Lister les termes GO | ✅ |
| `GET` | `/ontology/tree/{term_id}` | Arbre des enfants d'un terme | ✅ |
| `GET` | `/ontology/annotations` | Annotations d'un gène | ✅ |

---

## AI (`/api/v1/ai`)

Intégration avec Ollama pour l'interprétation IA.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/ai/chat` | Conversation IA (SSE streaming) | ✅ |
| `GET` | `/ai/conversations/{id}` | Historique d'une conversation | ✅ |
| `DELETE` | `/ai/conversations/{id}` | Supprimer une conversation | ✅ |
| `GET` | `/ai/interpretation/{analysis_id}` | Interprétation automatique | ✅ |
| `GET` | `/ai/quota` | Quota AI restant | ✅ |

---

## Billing (`/api/v1/billing`)

Gestion des abonnements Stripe.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/billing/checkout` | Créer session checkout Stripe | ✅ |
| `GET` | `/billing/subscription` | État actuel de l'abonnement | ✅ |
| `POST` | `/billing/customer-portal` | Session portal client Stripe | ✅ |

---

## Bookmarks (`/api/v1/bookmarks`)

Sauvegarde de sélections d'utilisateurs.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/bookmarks` | Créer un bookmark | ✅ |
| `GET` | `/bookmarks` | Liste des bookmarks | ✅ |
| `PATCH` | `/bookmarks/{id}` | Mettre à jour un bookmark | ✅ (owner) |
| `DELETE` | `/bookmarks/{id}` | Supprimer un bookmark | ✅ (owner) |

---

## Comments (`/api/v1/comments`)

Système de commentaires sur les projets.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/comments` | Ajouter un commentaire | ✅ |
| `GET` | `/comments?project_id={id}` | Commentaires d'un projet | ✅ |
| `PATCH` | `/comments/{id}` | Modifier un commentaire | ✅ (owner) |
| `DELETE` | `/comments/{id}` | Supprimer un commentaire | ✅ (owner) |

---

## History (`/api/v1/history`)

Historique des actions utilisateur.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/history` | Historique de l'utilisateur | ✅ |
| `GET` | `/history/project/{id}` | Historique d'un projet | ✅ |

---

## Integrations (`/api/v1/integrations`)

Intégrations avec des bases de données externes.

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/integrations/uniprot/{gene}` | Données UniProt d'un gène | ✅ |
| `GET` | `/integrations/ncbi/{gene}` | Données NCBI d'un gène | ✅ |

---

## Admin (`/api/v1/admin`)

Endpoints administrateur (rôle admin requis).

| Méthode | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/admin/users` | Liste des utilisateurs | ✅ (admin) |
| `GET` | `/admin/projects` | Liste de tous les projets | ✅ (admin) |
| `PATCH` | `/admin/users/{id}/role` | Changer le rôle d'un utilisateur | ✅ (admin) |
| `GET` | `/admin/stats` | Statistiques système | ✅ (admin) |
| `GET` | `/admin/ai-logs` | Logs d'utilisation IA | ✅ (admin) |

---

## Stripe Webhooks (`/api/v1/stripe/webhook`)

Endpoints internes pour les webhooks Stripe.

| Méthode | Endpoint | Description |
|---|---|---|
| `POST` | `/stripe/webhook` | Webhook Stripe (checkout, subscription, invoice) |

> ⚠️ **Non exposé dans le Swagger** — protégé par signature webhook Stripe.

---

## Schémas de réponse communs

### Pagination
```json
{
  "items": [...],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "total_pages": 3
}
```

### Erreur standard
```json
{
  "detail": "Message d'erreur"
}
```

### Validation error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Authentification

Tous les endpoints (sauf `/health`, `/`) nécessitent un token JWT Supabase dans l'en-tête :

```
Authorization: Bearer <supabase_jwt_token>
```

Les rôles sont vérifiés via le claim `role` du JWT :
- `user` — accès standard
- `admin` — accès admin + user standard