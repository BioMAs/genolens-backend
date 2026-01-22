# Correction : Erreur de Session SQLAlchemy avec l'AI Interpreter

## Problème Identifié

### Symptôme
Lors de l'utilisation de l'endpoint d'interprétation IA (), l'erreur suivante se produisait :

```
Erreur lors de la réponse: Object '<User at 0x740bc8871f90>' is already attached to session '93' (this is '92')
```

### Cause Racine

Le problème venait de la gestion des sessions SQLAlchemy dans un contexte asynchrone avec un appel long :

1. L'objet `User` était récupéré via les dépendances FastAPI (`require_ai_access`, `check_ai_quota`)
2. Ces dépendances attachaient l'objet `User` à une session SQLAlchemy
3. L'appel à Ollama pour générer l'interprétation prenait plusieurs minutes
4. Pendant ce temps, le contexte de session changeait
5. Quand on essayait d'utiliser cet objet `User` dans `increment_ai_usage()`, SQLAlchemy détectait qu'il était attaché à une ancienne session

### Code Problématique

**Fichier** : `app/api/deps/subscription.py`
**Fonction** : `increment_ai_usage()`
**Ligne** : ~151

```python
# AVANT (problématique)
db.add(user)  # Erreur : user est attaché à une autre session
await db.commit()
```

## Solution Appliquée

### Modification du Code

**Fichier** : `app/api/deps/subscription.py`
**Ligne** : ~151

```python
# APRÈS (corrigé)
user = await db.merge(user)  # Réattache user à la session actuelle
await db.commit()
```

### Explication de `db.merge()`

La méthode `merge()` de SQLAlchemy :
- Prend un objet qui peut être attaché à une autre session
- Crée une copie de cet objet dans la session actuelle
- Retourne la nouvelle instance attachée à la session courante
- Préserve toutes les modifications apportées à l'objet

C'est la solution recommandée pour gérer des objets qui traversent plusieurs contextes de session.

## Alternatives Considérées

### Option 1 : Recharger l'objet User
```python
# Re-query user dans la session actuelle
user = await db.get(User, user.id)
db.add(user)
```
**Inconvénient** : Nécessite une requête supplémentaire

### Option 2 : Utiliser expunge()
```python
# Détacher de l'ancienne session puis ajouter
db.expunge(user)
db.add(user)
```
**Inconvénient** : Plus complexe et peut causer des problèmes

### Option 3 : Passer l'ID au lieu de l'objet
Modifier `increment_ai_usage()` pour accepter `user_id` au lieu de `user`
**Inconvénient** : Modification plus invasive de l'API

## Tests de Validation

### Avant la Correction
```bash
curl -X POST https://api-v2.genolens.com/api/v1/datasets/{id}/comparisons/{name}/interpret \
  -H Authorization: Bearer {token}

# Résultat : Erreur 500 après génération de l'interprétation
# Message : Object <User at ...> is already attached to session...
```

### Après la Correction
```bash
curl -X POST https://api-v2.genolens.com/api/v1/datasets/{id}/comparisons/{name}/interpret \
  -H Authorization: Bearer {token}

# Résultat : 200 OK
# Réponse : { interpretation: ..., cached: false, ... }
```

## Impact

### Fichiers Modifiés
- `app/api/deps/subscription.py` (1 ligne modifiée)

### Services Affectés
- Endpoint d'interprétation IA : `POST /api/v1/datasets/{dataset_id}/comparisons/{comparison_name}/interpret`
- Endpoint de questions IA : `POST /api/v1/datasets/{dataset_id}/comparisons/{comparison_name}/ask` (si utilisé)

### Compatibilité
- ✅ Pas de changement dans l'API publique
- ✅ Pas de migration de base de données requise
- ✅ Compatible avec toutes les versions existantes

## Prévention Future

### Bonnes Pratiques SQLAlchemy Async

1. **Toujours utiliser `db.merge()`** pour les objets qui traversent plusieurs contextes
2. **Éviter de garder des références** à des objets ORM après des opérations longues
3. **Privilégier les IDs** pour passer des références entre contextes
4. **Utiliser `await db.refresh()`** pour recharger un objet dans la session actuelle

### Pattern Recommandé pour les Opérations Longues

```python
async def long_operation_endpoint(
    user: Annotated[User, Depends(get_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    # Sauvegarder l'ID avant l'opération longue
    user_id = user.id
    
    # Opération longue (ex: appel Ollama)
    result = await long_async_operation()
    
    # Recharger l'objet dans la session actuelle
    user = await db.get(User, user_id)
    # Ou utiliser merge si on a modifié user
    # user = await db.merge(user)
    
    # Maintenant on peut utiliser user en toute sécurité
    user.some_field += 1
    await db.commit()
```

## Date de Correction
- **Date** : 2026-01-22
- **Version** : GenoLens Next v1.0.0
- **Commit** : (à compléter après commit git)

## Références
- SQLAlchemy Session Basics : https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- merge() documentation : https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.Session.merge
