"""
AnnoDbService — catalogue des bases de données disponibles dans les fichiers anno.db.

Lit le fichier JSON pré-généré (app/data/anno_db_categories.json) et expose
une méthode synchrone get_categories(species) pour l'utiliser dans les endpoints.

Pour regénérer le JSON après mise à jour des fichiers anno.db :
    Rscript backend/r_scripts/list_anno_db_categories.R \\
        /path/to/anno.db \\
        backend/app/data/anno_db_categories.json
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CATEGORIES_FILE = Path(__file__).parent.parent / "data" / "anno_db_categories.json"

# Supported species: value in anno.db filename → display name
SUPPORTED_SPECIES: dict[str, str] = {
    "human":     "Homo sapiens (Human)",
    "mouse":     "Mus musculus (Mouse)",
    "rat":       "Rattus norvegicus (Rat)",
    "zebrafish": "Danio rerio (Zebrafish)",
    "pig":       "Sus scrofa (Pig)",
}

# Species name aliases accepted from the Project.species field → canonical key
SPECIES_ALIASES: dict[str, str] = {
    "human":       "human",
    "homo_sapiens": "human",
    "hg38":        "human",
    "hg19":        "human",
    "mouse":       "mouse",
    "mus_musculus": "mouse",
    "mm10":        "mouse",
    "mm39":        "mouse",
    "rat":         "rat",
    "rattus_norvegicus": "rat",
    "rn6":         "rat",
    "rn7":         "rat",
    "zebrafish":   "zebrafish",
    "danio_rerio": "zebrafish",
    "drerio":      "zebrafish",
    "pig":         "pig",
    "sus_scrofa":  "pig",
}

# In-memory cache (loaded once at first call)
_cache: Optional[dict[str, list[str]]] = None


def _load_cache() -> dict[str, list[str]]:
    global _cache
    if _cache is None:
        if not _CATEGORIES_FILE.exists():
            logger.warning(f"anno_db_categories.json not found at {_CATEGORIES_FILE}. Returning empty catalogue.")
            _cache = {}
        else:
            with open(_CATEGORIES_FILE, "r") as f:
                data = json.load(f)
            # Strip the metadata comment key
            _cache = {k: v for k, v in data.items() if not k.startswith("_")}
            logger.info(f"Loaded anno_db_categories for species: {list(_cache.keys())}")
    return _cache


def resolve_species(species: str) -> Optional[str]:
    """Normalize a species string to the canonical key (e.g. 'hg38' → 'human')."""
    return SPECIES_ALIASES.get(species.lower())


def get_categories(species: str) -> list[str]:
    """
    Return the list of annotation database categories available for the given species.

    Args:
        species: Species string (canonical or alias, e.g. 'human', 'hg38', 'mouse').

    Returns:
        List of category names, or empty list if species not found.
    """
    canonical = resolve_species(species)
    if canonical is None:
        return []
    catalogue = _load_cache()
    return catalogue.get(canonical, [])
