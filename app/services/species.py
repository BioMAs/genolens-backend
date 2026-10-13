"""
The organism a dataset is about, and its NCBI taxon for databases keyed that way (STRING).

The analysis wizard asks for the species and stores it in ``SelfServiceAnalysis.params``. The
DEG datasets the analysis produces only point back to it through ``dataset_metadata
['analysis_id']``, so no client can read the species — the interaction network used to guess it
back from gene names, and took a rat project keyed by symbols for mouse.
"""
import re
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Dataset, SelfServiceAnalysis

# Every spelling seen for the organisms the wizard offers: common names (what it stores),
# scientific names, their Ensembl/Bioconductor contractions, and genome builds — the same
# vocabulary functional_enrichment.R accepts.
TAXON_BY_SPECIES: dict[str, int] = {
    "human": 9606,
    "homo sapiens": 9606,
    "hsapiens": 9606,
    "hg19": 9606,
    "hg38": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "mmusculus": 10090,
    "mm10": 10090,
    "mm39": 10090,
    "rat": 10116,
    "rattus norvegicus": 10116,
    "rnorvegicus": 10116,
    "rn6": 10116,
    "rn7": 10116,
    "zebrafish": 7955,
    "danio rerio": 7955,
    "drerio": 7955,
    "pig": 9823,
    "sus scrofa": 9823,
    "sscrofa": 9823,
    "medaka": 8090,
    "oryzias latipes": 8090,
    "olatipes": 8090,
}

_SEPARATORS = re.compile(r"[\s_.]+")


def species_taxon(species: object) -> Optional[int]:
    """NCBI taxon of a species name or id, ``None`` when it is not one we know.

    ``None`` rather than human: the caller decides what an unknown organism falls back to.
    """
    if isinstance(species, bool):
        return None
    if isinstance(species, int):
        return species if species > 0 else None
    if not isinstance(species, str):
        return None
    key = _SEPARATORS.sub(" ", species.strip().lower())
    if key.isdigit():
        return int(key) or None
    return TAXON_BY_SPECIES.get(key)


async def declared_species(db: AsyncSession, dataset: Dataset) -> Optional[str]:
    """The species recorded for a dataset, ``None`` when nothing records one.

    The dataset's own metadata first, then the parameters of the analysis that produced it.
    """
    metadata = dataset.dataset_metadata or {}
    if metadata.get("species"):
        return str(metadata["species"])
    analysis_id = metadata.get("analysis_id")
    if analysis_id:
        try:
            analysis = await db.get(SelfServiceAnalysis, UUID(str(analysis_id)))
        except (ValueError, TypeError):
            analysis = None
        if analysis is not None and (analysis.params or {}).get("species"):
            return str(analysis.params["species"])
    return None
