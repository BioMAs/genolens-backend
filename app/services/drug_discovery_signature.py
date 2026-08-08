"""Construction d'une signature Drug Discovery depuis une comparaison de l'utilisateur.

Tout ce qui est **notre** logique vit ici, pour que `api/endpoints/drug_discovery.py` reste le
passthrough que son docstring promet. Le service amont (`genolens-dd`) attend une signature
`{condition -> symboles}` plus un effectif de réplicats par condition ; une comparaison
GenoLens fournit des DEG marqués `UP`/`DOWN`. Traduire l'un vers l'autre pose trois questions,
et chacune a une mauvaise réponse évidente :

1. **Une condition ou deux ?** Une seule pseudo-condition tenant tous les DEG forcerait un
   effectif `n1 + n2`, qui franchirait la porte SIG001/SIG002 conçue pour attraper les bras
   sous-dimensionnés. Deux conditions coûtent zéro statistiquement — `known_target_route`
   prend l'union — et achètent une porte de puissance correcte.
2. **Combien de réplicats ?** Jamais devinés. Une valeur par défaut « raisonnable » est
   exactement ce que la règle SIG005 refuse : « supposer qu'il est suffisant serait la valeur
   par défaut la plus dangereuse du lot ».
3. **Combien de gènes ?** Plafonnés, et la troncature est **dite**. Une signature de 3 000
   gènes sur un univers de ~15 000 fait dégénérer le percentile moyen vers 0,5, et le tirage
   apparié amont tronque en silence quand une strate est trop petite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    SelfServiceAnalysis,
)
from app.services.data_processor import data_processor
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

#: Séparateurs acceptés pour deviner les conditions depuis un nom de comparaison.
#:
#: **Volontairement restreint à `vs`.** `data_processor._parse_comparison_name` accepte aussi
#: `_`, `-` et `.` nus ; réutilisé ici, il découperait `Treated_24h_vs_Ctrl_24h` en morceaux
#: qui ne sont pas des conditions. Un nom de condition faux n'est pas cosmétique : c'est la clé
#: du dictionnaire de réplicats, donc une porte de puissance appliquée au mauvais bras.
_VS_SEPARATOR = re.compile(r"_vs_|_VS_|-vs-|-VS-", re.IGNORECASE)

#: Un identifiant Ensembl là où on attend un symbole.
_ENSEMBL_ID = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)

#: Au-delà de cette proportion d'identifiants Ensembl, prévenir avant de brûler un appel amont.
_ENSEMBL_WARNING_RATIO = 0.8

Direction = Literal["UP", "DOWN"]
DirectionFilter = Literal["both", "up", "down"]

ReplicatesSource = Literal[
    "analysis_samplesheet", "project_samplesheet", "user", "unknown"
]


@dataclass(frozen=True)
class SignatureCondition:
    """Un bras de la comparaison, prêt à être déposé."""

    name: str
    direction: Direction
    genes: tuple[str, ...]
    #: Nombre de DEG disponibles **avant** plafonnement. Distinct de `len(genes)` pour que
    #: « 1 000 gènes envoyés » ne se lise jamais « il n'y en avait que 1 000 ».
    n_available: int
    truncated: bool
    replicates: Optional[int]
    replicates_source: ReplicatesSource


@dataclass(frozen=True)
class SignaturePayload:
    """Signature construite, avec ce qui doit être montré avant de lancer le calcul."""

    conditions: tuple[SignatureCondition, ...]
    species: Optional[str]
    warnings: tuple[str, ...]

    @property
    def needs_replicates(self) -> bool:
        return any(c.replicates is None for c in self.conditions)

    @property
    def genes_sent_total(self) -> int:
        return sum(len(c.genes) for c in self.conditions)

    def genes_by_condition(self) -> dict[str, list[str]]:
        return {c.name: list(c.genes) for c in self.conditions}

    def replicates(self) -> dict[str, int]:
        return {
            c.name: c.replicates for c in self.conditions if c.replicates is not None
        }


def parse_condition_names(comparison_name: str) -> Optional[tuple[str, str]]:
    """`(condition1, condition2)` déduits du nom, ou `None` si le nom ne le dit pas.

    Ne devine que sur un `vs` explicite. Rendre `None` est un résultat utile : l'appelant
    demandera les noms à l'utilisateur, ce qui vaut mieux que deux étiquettes inventées et
    utilisées comme clés d'un dictionnaire de réplicats.
    """
    parts = _VS_SEPARATOR.split(comparison_name)
    if len(parts) != 2:
        return None
    first, second = parts[0].strip(), parts[1].strip()
    if not first or not second:
        return None
    return first, second


async def resolve_condition_names(
    db: AsyncSession, dataset: Dataset, comparison_name: str
) -> tuple[str, str]:
    """Noms des deux bras, par ordre de fiabilité décroissante.

    1. Le dataset METADATA_CONTRAST de l'analyse — ce sont les noms que le script R a
       réellement utilisés (`condition1` = numérateur), donc la seule source autoritative.
    2. Une découpe sur `vs` dans le nom de la comparaison.
    3. `up` / `down`, à charge pour l'UI de faire renommer.
    """
    metadata = dataset.dataset_metadata or {}
    analysis_id = metadata.get("analysis_id")
    if analysis_id:
        names = await _condition_names_from_contrast_dataset(
            db, str(analysis_id), comparison_name
        )
        if names is not None:
            return names

    parsed = parse_condition_names(comparison_name)
    if parsed is not None:
        return parsed

    return "up", "down"


async def _condition_names_from_contrast_dataset(
    db: AsyncSession, analysis_id: str, comparison_name: str
) -> Optional[tuple[str, str]]:
    """Lit la table de contrastes de l'analyse. Rend `None` plutôt que de lever.

    Une comparaison reste analysable sans sa table de contrastes ; échouer ici priverait
    l'utilisateur du module pour une raison qui n'est pas la sienne.
    """
    try:
        analysis = await db.get(SelfServiceAnalysis, UUID(str(analysis_id)))
    except (ValueError, TypeError):
        return None
    if analysis is None or analysis.comparisons_dataset_id is None:
        return None

    frame = await _load_dataset_frame(db, analysis.comparisons_dataset_id)
    if frame is None:
        return None

    columns = {str(c).strip().lower(): c for c in frame.columns}
    # Mêmes alias que `run_multimethod_pipeline.R` : la lecture doit accepter exactement ce
    # que le pipeline a accepté, sinon les deux divergent sur un fichier valide.
    comp_col = _first_present(
        columns, ("comparison", "comparison_id", "comparison_name", "name", "contrast")
    )
    cond1_col = _first_present(columns, ("condition1", "test", "numerator", "group1"))
    cond2_col = _first_present(columns, ("condition2", "ref", "denominator", "group2"))
    if not all((comp_col, cond1_col, cond2_col)):
        return None

    for _, row in frame.iterrows():
        if str(row[comp_col]).strip() != comparison_name:
            continue
        first, second = str(row[cond1_col]).strip(), str(row[cond2_col]).strip()
        if first and second:
            return first, second
    return None


def _first_present(columns: dict[str, Any], candidates: tuple[str, ...]) -> Optional[Any]:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


async def _load_dataset_frame(db: AsyncSession, dataset_id: UUID) -> Optional[Any]:
    """Charge un petit dataset tabulaire (samplesheet, contrastes) en DataFrame."""
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or not dataset.parquet_file_path:
        return None
    try:
        raw = await storage_service.download_file(dataset.parquet_file_path)
        return await data_processor.get_dataframe(raw)
    except Exception:  # noqa: BLE001 - lecture best-effort, jamais bloquante
        logger.warning("Could not read dataset %s for signature build", dataset_id, exc_info=True)
        return None


async def resolve_replicates(
    db: AsyncSession, dataset: Dataset, condition_names: tuple[str, str]
) -> tuple[dict[str, int], ReplicatesSource]:
    """Effectifs par condition, ou `({}, "unknown")`. **Jamais une supposition.**

    Échelon 1, le samplesheet de l'analyse, avec une soustraction qui n'est pas un détail :
    les échantillons **retirés au QC** ne figuraient pas dans le modèle différentiel. Les
    compter pourrait pousser un vrai bras à n=2 au-delà de la porte SIG002 — soit exactement
    l'inverse de ce que cette porte protège.

    Échelon 2, un samplesheet du projet dont la colonne de condition contient les deux noms.
    Plusieurs candidats ⇒ non résolu : choisir serait deviner.
    """
    metadata = dataset.dataset_metadata or {}
    analysis_id = metadata.get("analysis_id")

    if analysis_id:
        counts = await _replicates_from_analysis(db, str(analysis_id), condition_names)
        if counts:
            return counts, "analysis_samplesheet"

    counts = await _replicates_from_project(db, dataset, condition_names)
    if counts:
        return counts, "project_samplesheet"

    return {}, "unknown"


async def _replicates_from_analysis(
    db: AsyncSession, analysis_id: str, condition_names: tuple[str, str]
) -> dict[str, int]:
    try:
        analysis = await db.get(SelfServiceAnalysis, UUID(str(analysis_id)))
    except (ValueError, TypeError):
        return {}
    if analysis is None or analysis.samples_dataset_id is None:
        return {}

    frame = await _load_dataset_frame(db, analysis.samples_dataset_id)
    if frame is None:
        return {}

    removed = await _qc_removed_sample_ids(db, analysis)
    return _count_by_condition(frame, condition_names, removed)


async def _qc_removed_sample_ids(
    db: AsyncSession, analysis: SelfServiceAnalysis
) -> set[str]:
    """Échantillons écartés par le QC du pipeline R, donc absents du modèle différentiel."""
    vst_id = (analysis.intermediate_dataset_ids or {}).get("vst")
    if not vst_id:
        return set()
    try:
        vst = await db.get(Dataset, UUID(str(vst_id)))
    except (ValueError, TypeError):
        return set()
    if vst is None:
        return set()
    report = (vst.dataset_metadata or {}).get("qc_report") or {}
    return {str(s).strip() for s in report.get("removed_sample_ids", []) if str(s).strip()}


async def _replicates_from_project(
    db: AsyncSession, dataset: Dataset, condition_names: tuple[str, str]
) -> dict[str, int]:
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == dataset.project_id,
            Dataset.type == DatasetType.METADATA_SAMPLE,
            Dataset.status == DatasetStatus.READY,
        )
    )
    matches: list[dict[str, int]] = []
    for candidate in result.scalars().all():
        frame = await _load_dataset_frame(db, candidate.id)
        if frame is None:
            continue
        counts = _count_by_condition(frame, condition_names, removed=set())
        if len(counts) == 2:
            matches.append(counts)

    # Plusieurs samplesheets décrivent les deux conditions : on ne choisit pas. L'utilisateur
    # saura lequel est le bon, nous non — et un mauvais effectif fausse la porte de puissance.
    if len(matches) != 1:
        return {}
    return matches[0]


def _count_by_condition(
    frame: Any, condition_names: tuple[str, str], removed: set[str]
) -> dict[str, int]:
    columns = {str(c).strip().lower(): c for c in frame.columns}
    sample_col = _first_present(
        columns, ("sample_id", "sample", "ini.sample.name", "sample_name")
    )
    condition_col = _first_present(columns, ("condition", "group", "treatment"))
    if condition_col is None:
        return {}

    counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        if sample_col is not None and str(row[sample_col]).strip() in removed:
            continue
        value = str(row[condition_col]).strip()
        if value in condition_names:
            counts[value] = counts.get(value, 0) + 1

    if len(counts) != 2:
        return {}
    return counts


async def resolve_species(db: AsyncSession, dataset: Dataset) -> str:
    """Espèce du dataset, humain par défaut.

    Même échelle de résolution que l'enrichissement d'intersection.
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
    return "human"


async def build_signature_payload(
    db: AsyncSession,
    dataset: Dataset,
    comparison_name: str,
    *,
    padj_max: float,
    logfc_min: float,
    directions: DirectionFilter = "both",
    max_genes_per_condition: int = 1000,
    replicates_override: Optional[dict[str, int]] = None,
) -> SignaturePayload:
    """Assemble la signature à déposer, sans appeler le service amont.

    Une **direction vide est omise**, jamais envoyée avec zéro gène : la règle SIG004 refuse
    une condition vide, et à raison — un enrichissement sur l'ensemble vide rendrait des
    p-values dénuées de sens plutôt qu'une erreur. Une comparaison à sens unique dépose donc
    une seule condition.
    """
    # Import tardif : `datasets.py` importe déjà beaucoup, et l'importer au chargement
    # créerait un cycle avec les endpoints qui utiliseront ce service.
    from app.api.endpoints.datasets import select_deg_genes

    name_up, name_down = await resolve_condition_names(db, dataset, comparison_name)
    counts, source = await resolve_replicates(db, dataset, (name_up, name_down))
    if replicates_override:
        counts = {**counts, **replicates_override}
        source = "user"

    wanted: list[tuple[str, Direction]] = []
    if directions in ("both", "up"):
        wanted.append((name_up, "UP"))
    if directions in ("both", "down"):
        wanted.append((name_down, "DOWN"))

    conditions: list[SignatureCondition] = []
    warnings: list[str] = []
    for name, direction in wanted:
        rows = await select_deg_genes(
            db, dataset.id, comparison_name,
            regulation=direction, padj_max=padj_max, logfc_min=logfc_min,
        )
        # `gene_name or gene_id` : même règle d'identité que l'analyse de Venn. Dédupliqué,
        # parce que plusieurs identifiants Ensembl peuvent porter le même symbole et qu'un
        # doublon gonflerait artificiellement la taille de la signature.
        symbols: list[str] = []
        seen: set[str] = set()
        for row in rows:
            symbol = (row.gene_name or row.gene_id or "").strip()
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

        if not symbols:
            continue

        truncated = len(symbols) > max_genes_per_condition
        conditions.append(
            SignatureCondition(
                name=name,
                direction=direction,
                genes=tuple(symbols[:max_genes_per_condition]),
                n_available=len(symbols),
                truncated=truncated,
                replicates=counts.get(name),
                replicates_source=source if counts.get(name) is not None else "unknown",
            )
        )
        if truncated:
            warnings.append(
                f"{name}: {len(symbols)} genes match the thresholds, the "
                f"{max_genes_per_condition} most significant were kept."
            )

    species = await resolve_species(db, dataset)
    if species.lower() not in {"human", "homo sapiens", "hsapiens"}:
        # Averti, pas bloqué : c'est le choix de l'utilisateur, mais le classement amont est
        # humain (TCGA), donc des symboles murins tomberaient presque tous en non résolus.
        warnings.append(
            f"This comparison is annotated as '{species}'. Drug Discovery rankings are "
            "human (TCGA); most non-human symbols will not resolve."
        )

    if _looks_like_ensembl_ids(conditions):
        warnings.append(
            "Most gene identifiers look like Ensembl IDs rather than symbols. Drug Discovery "
            "resolves symbols, so few will match — check that your matrix carried a gene "
            "symbol column."
        )

    if not conditions:
        warnings.append(
            "No gene passes these thresholds in either direction. Loosen padj or |log2FC|."
        )

    return SignaturePayload(
        conditions=tuple(conditions), species=species, warnings=tuple(warnings)
    )


def _looks_like_ensembl_ids(conditions: list[SignatureCondition]) -> bool:
    """Le script R met `gene_name = gene_id` quand la matrice n'a pas de colonne symbole.

    Le détecter ici évite de déposer une signature qui ne résoudra pas, et de rendre à
    l'utilisateur un « aucun gène résolu » qui ressemble à une panne.
    """
    genes = [gene for condition in conditions for gene in condition.genes]
    if not genes:
        return False
    ensembl = sum(1 for gene in genes if _ENSEMBL_ID.match(gene))
    return ensembl / len(genes) >= _ENSEMBL_WARNING_RATIO
