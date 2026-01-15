# Scripts SQL Supabase - GenoLens v2

## 🎯 Quel script utiliser ?

### ⭐ Installation complète (RECOMMANDÉ)

**Utilisez ce script pour une nouvelle installation :**

```
supabase_schema_v2_final.sql
```

**Caractéristiques :**
- ✅ Crée toutes les tables nécessaires (projects, datasets, deg_genes, project_members)
- ✅ Gère les conflits d'enums (utilise `user_role_v2`)
- ✅ Désactive temporairement les triggers
- ✅ Compatible avec bases existantes (legacy app)
- ✅ Messages de confirmation détaillés

**Contenu :**
- Tables de base (projects, datasets, samples, project_members)
- Table `deg_genes` pour performance DEG
- Row Level Security (RLS)
- Indexes de performance
- Triggers updated_at

---

### 🚀 Ajout de l'optimisation Enrichment

**Après avoir exécuté le script principal, ajoutez :**

```
supabase_enrichment_pathways_schema.sql
```

**Caractéristiques :**
- ✅ Table `enrichment_pathways` avec 8 index
- ✅ Performance 20-50x plus rapide
- ✅ Support des formats clusterProfiler, DAVID, GSEA
- ✅ RLS policies automatiques

**⚠️ Prérequis :** `supabase_schema_v2_final.sql` doit être exécuté avant

---

## 📋 Liste complète des scripts

| Script | Description | Quand l'utiliser |
|--------|-------------|------------------|
| **`supabase_schema_v2_final.sql`** ⭐ | Script complet + gestion conflits | **TOUJOURS** - Installation initiale |
| **`supabase_enrichment_pathways_schema.sql`** | Table enrichment_pathways | Après le script principal |
| `supabase_schema_safe.sql` | Version alternative avec triggers désactivés | Si le script final échoue |
| `supabase_complete_schema.sql` | Version sans gestion de conflits | Base vierge uniquement |
| `supabase_deg_genes_schema.sql` | Table deg_genes uniquement | ❌ Déprécié (inclus dans v2_final) |

---

## 🔄 Ordre d'exécution

### Installation nouvelle base

1. **`supabase_schema_v2_final.sql`** - Tables de base + deg_genes
2. **`supabase_enrichment_pathways_schema.sql`** - Optimisation enrichments

### Mise à jour base existante

Si vous avez déjà exécuté un ancien script :

1. Vérifiez les tables existantes :
   ```sql
   SELECT table_name FROM information_schema.tables
   WHERE table_schema = 'public'
   AND table_name IN ('projects', 'datasets', 'deg_genes', 'enrichment_pathways')
   ORDER BY table_name;
   ```

2. Si `deg_genes` existe mais pas `enrichment_pathways` :
   - Exécutez uniquement `supabase_enrichment_pathways_schema.sql`

3. Si aucune table n'existe :
   - Exécutez `supabase_schema_v2_final.sql` puis `supabase_enrichment_pathways_schema.sql`

---

## 📖 Documentation associée

- **Installation complète** : [/docs/SUPABASE_SETUP.md](../../docs/SUPABASE_SETUP.md)
- **Optimisation Enrichment** : [/docs/ENRICHMENT_PATHWAYS_SETUP.md](../../docs/ENRICHMENT_PATHWAYS_SETUP.md)
- **Dépannage** : [/docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)
- **Architecture** : [/docs/DATABASE_ARCHITECTURE.md](../../docs/DATABASE_ARCHITECTURE.md)

---

## ⚠️ Notes importantes

### Coexistence avec application legacy

Tous les scripts créent de **NOUVELLES tables** pour GenoLens v2 :
- `projects` (≠ `sequencing_projects`)
- `datasets`
- `deg_genes`
- `enrichment_pathways`
- `project_members`

**Les tables de votre application legacy ne sont PAS modifiées.**

### Gestion des erreurs courantes

#### Erreur : "invalid input value for enum user_role"
→ Utilisez `supabase_schema_v2_final.sql` qui crée `user_role_v2`

#### Erreur : "column 'name' does not exist"
→ Triggers existants - Le script `v2_final` les désactive automatiquement

#### Erreur : "relation already exists"
→ Normal si réexécution - Le script vérifie l'existence avant création

---

## 🧪 Vérification après installation

```sql
-- 1. Vérifier les tables créées
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('projects', 'datasets', 'deg_genes', 'enrichment_pathways')
ORDER BY table_name;
-- Résultat attendu: 4 tables

-- 2. Vérifier les index deg_genes
SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'deg_genes';
-- Résultat attendu: 7

-- 3. Vérifier les index enrichment_pathways
SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'enrichment_pathways';
-- Résultat attendu: 8

-- 4. Vérifier les RLS policies
SELECT tablename, COUNT(*) as policy_count
FROM pg_policies
WHERE tablename IN ('projects', 'datasets', 'deg_genes', 'enrichment_pathways')
GROUP BY tablename;
-- Chaque table doit avoir des policies
```

---

## 📞 Support

En cas de problème :
1. Consultez [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)
2. Vérifiez les logs Supabase
3. Contactez l'équipe technique

---

**Dernière mise à jour** : 2025-12-29
**Version** : GenoLens v2.0
