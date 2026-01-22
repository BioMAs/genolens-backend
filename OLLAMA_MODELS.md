# Modèles Ollama Installés

## Modèles Disponibles

### llama3.2:3b (Par défaut)
- **Taille** : 2.0 GB
- **Usage** : Modèle par défaut pour l'interprétation biologique
- **Performance** : Léger, rapide, bon pour l'interprétation basique
- **Config** : Utilisé par défaut dans `app/services/ai_interpreter.py`

### llama2:latest
- **Taille** : 3.8 GB
- **Usage** : Alternative plus puissante
- **Performance** : Plus gourmand en ressources mais plus précis

## Commandes Utiles

### Lister les modèles
```bash
docker compose -f docker-compose.prod.yml exec ollama ollama list
```

### Télécharger un nouveau modèle
```bash
docker compose -f docker-compose.prod.yml exec ollama ollama pull <model-name>
```

### Tester un modèle
```bash
docker compose -f docker-compose.prod.yml exec ollama ollama run llama3.2:3b Test prompt
```

### Supprimer un modèle
```bash
docker compose -f docker-compose.prod.yml exec ollama ollama rm <model-name>
```

## Modèles Recommandés pour la Bioinformatique

- **llama3.2:3b** (2GB) - Par défaut, bon équilibre
- **biomistral** (7GB) - Spécialisé en biologie
- **llama3.1:8b** (8GB) - Usage général avancé
- **mixtral:8x7b** (26GB) - Haute qualité mais très gourmand

## Espace Disque

Volume Ollama : `backend_ollama_data`
- Actuellement utilisé : ~6 GB (llama3.2:3b + llama2)
- Emplacement : `/var/lib/docker/volumes/backend_ollama_data`

## Configuration API

Le modèle par défaut est configuré dans :
- Fichier : `app/services/ai_interpreter.py`
- Ligne : ~205
- Variable : `model: str = "llama3.2:3b"`

Pour changer le modèle par défaut, modifier cette ligne et redémarrer l'API.
