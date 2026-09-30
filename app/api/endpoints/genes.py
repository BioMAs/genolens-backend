"""
Recherche de gènes transverse : « où ce gène apparaît-il dans mes comparaisons ? ».

Interroge `deg_genes`, la table des statistiques par gène que remplit l'ingestion d'un
dataset DEG et que lit la table des DEG d'une comparaison. Une ligne de résultat est un
gène DANS une comparaison, avec les chiffres que la page de comparaison affichera.

L'ancienne version ne cherchait pas de gènes : elle comparait le texte tapé aux noms et
descriptions de datasets, puis renvoyait ce texte en majuscules comme s'il s'agissait d'un
gène trouvé. Rien ici ne doit plus fabriquer un résultat que la base ne contient pas.
"""

from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import String, and_, case, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.project_access import accessible_projects_clause
from app.core.supabase_auth import SupabaseUser
from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    DegGene,
    Project,
    SelfServiceAnalysis,
)

router = APIRouter(prefix="/genes", tags=["genes"])

# Une palette de commandes en montre six ; au-delà de cinquante, ce n'est plus une
# recherche mais un export, et la table des DEG existe pour cela.
DEFAULT_LIMIT = 8
MAX_LIMIT = 50

# Un seul caractère ferait correspondre un gène sur vingt-six de chaque comparaison
# accessible — un balayage, pas une recherche.
MIN_QUERY_LENGTH = 2


class GeneSearchResult(BaseModel):
    """Un gène dans une comparaison."""

    gene_id: str
    # Le symbole quand l'ingestion en a trouvé un, sinon l'identifiant : c'est ce qui
    # s'affiche, et un gène sans symbole doit rester nommable.
    gene_symbol: str
    project_id: str
    project_name: str
    dataset_id: str
    # Présent seulement quand le dataset vient d'une analyse qui existe encore ; la page
    # de comparaison a alors une URL propre à l'analyse.
    analysis_id: Optional[str] = None
    analysis_name: Optional[str] = None
    comparison_name: str
    log_fc: Optional[float] = None
    padj: Optional[float] = None
    regulation: Optional[str] = None
    # Vrai quand le symbole ou l'identifiant est exactement la requête, faux pour un
    # simple préfixe : l'interface peut ainsi distinguer « TP53 » de « TP53BP1 ».
    exact: bool


class GeneSearchResponse(BaseModel):
    results: list[GeneSearchResult]
    total: int
    query: str


def _escape_like(text: str) -> str:
    """Neutralise les jokers de LIKE, pour que « _ » ou « % » tapés restent littéraux."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.get("/search", response_model=GeneSearchResponse)
async def search_genes(
    q: Annotated[
        str,
        Query(
            min_length=MIN_QUERY_LENGTH,
            max_length=100,
            description="Gene symbol or gene ID, or its beginning (e.g. TP53, ENSG00000141510)",
        ),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    project_id: Annotated[Optional[UUID], Query(description="Limit search to one project")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> GeneSearchResponse:
    """
    Search the stored DEG statistics for a gene, across every project the user owns or
    is a member of.

    Matching is case-insensitive on the gene symbol and on the gene ID. Genes whose
    symbol or ID *starts with* the query are returned; exact matches come first, then
    the rest, each ordered by adjusted p-value.

    Only READY DEG datasets are searched. A project the user cannot access contributes
    nothing — its genes are never matched, so the response does not reveal them.
    """
    needle = q.strip().upper()
    if len(needle) < MIN_QUERY_LENGTH:
        return GeneSearchResponse(results=[], total=0, query=q)

    pattern = _escape_like(needle) + "%"
    # Les deux expressions sont couvertes par des index `upper(...) text_pattern_ops`
    # (migration gene_search_indexes_001) ; les modifier ici les rendrait inutilisables.
    symbol_key = func.upper(DegGene.gene_name)
    id_key = func.upper(DegGene.gene_id)
    is_exact = or_(symbol_key == needle, id_key == needle)

    analysis_id_text = Dataset.dataset_metadata["analysis_id"].as_string()

    stmt = (
        select(
            DegGene.gene_id,
            DegGene.gene_name,
            DegGene.comparison_name,
            DegGene.log_fc,
            DegGene.padj,
            DegGene.regulation,
            Dataset.id.label("dataset_id"),
            Project.id.label("project_id"),
            Project.name.label("project_name"),
            SelfServiceAnalysis.id.label("analysis_id"),
            SelfServiceAnalysis.name.label("analysis_name"),
            case((is_exact, True), else_=False).label("exact"),
        )
        .join(Dataset, Dataset.id == DegGene.dataset_id)
        .join(Project, Project.id == Dataset.project_id)
        # Jointure externe : un dataset DEG téléversé n'a pas d'analyse, et une analyse
        # supprimée laisse un `analysis_id` orphelin dans les métadonnées. Comparer en
        # texte évite un CAST en uuid qui échouerait sur une valeur mal formée.
        .outerjoin(
            SelfServiceAnalysis,
            and_(
                SelfServiceAnalysis.project_id == Project.id,
                cast(SelfServiceAnalysis.id, String) == analysis_id_text,
            ),
        )
        .where(
            or_(symbol_key.like(pattern, escape="\\"), id_key.like(pattern, escape="\\")),
            Dataset.type == DatasetType.DEG,
            Dataset.status == DatasetStatus.READY,
            accessible_projects_clause(current_user.user_id),
        )
        .order_by(
            is_exact.desc(),
            DegGene.padj.asc().nulls_last(),
            func.coalesce(DegGene.gene_name, DegGene.gene_id),
            Project.name,
            DegGene.comparison_name,
        )
        .limit(limit)
    )
    if project_id:
        stmt = stmt.where(Project.id == project_id)

    rows = (await db.execute(stmt)).all()

    results = [
        GeneSearchResult(
            gene_id=row.gene_id,
            gene_symbol=row.gene_name or row.gene_id,
            project_id=str(row.project_id),
            project_name=row.project_name,
            dataset_id=str(row.dataset_id),
            analysis_id=str(row.analysis_id) if row.analysis_id else None,
            analysis_name=row.analysis_name,
            comparison_name=row.comparison_name,
            log_fc=row.log_fc,
            padj=row.padj,
            regulation=row.regulation,
            exact=bool(row.exact),
        )
        for row in rows
    ]
    return GeneSearchResponse(results=results, total=len(results), query=q)
