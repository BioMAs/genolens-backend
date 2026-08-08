"""Drug Discovery endpoints — thin passthrough to the genolens-dd service.

**EVERY endpoint here requires an authenticated user on a TEAM or ON_PREMISE plan.**
genolens-dd is closed by an API key precisely so its rankings are not world-readable. An
unauthenticated — or merely authenticated — route on this side would hand the same rankings to
anyone who can reach `api-v2.genolens.com`, re-opening the product through a different door while
the key gave the appearance of a closed one. That failure would look like a working feature, which
is why it is stated here rather than left to be noticed.

The handlers hold no business logic: genolens-dd owns the scoring contract, and a second
implementation of anything on this side is how the two services start disagreeing.
"""

import logging
from typing import Annotated, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.subscription import require_team_plan
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset, User
from app.services.drug_discovery import (
    DrugDiscoveryClient,
    DrugDiscoveryRejected,
    DrugDiscoveryUnavailable,
    get_drug_discovery_client,
)
from app.services.drug_discovery_signature import (
    SignaturePayload,
    build_signature_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drug-discovery", tags=["drug-discovery"])


class RunRequest(BaseModel):
    """Mirrors the genolens-dd payload. Not revalidated here — see the module docstring."""

    indication: Optional[str] = Field(
        default=None,
        description="TCGA project code, e.g. TCGA-BRCA. Null ranks pan-cancer.",
        examples=["TCGA-BRCA"],
    )
    profile: str = Field(
        default="default_oncology",
        description="Named weight profile resolved by genolens-dd.",
    )
    allow_excluded: bool = Field(
        default=False,
        description=(
            "Force a run on an indication excluded from the disease axis. The result then "
            "carries no disease axis at all and genolens-dd returns an explicit warning."
        ),
    )


def _reraise(exc: Exception) -> HTTPException:
    """Translate the client's two exception types into HTTP.

    Upstream 4xx detail is passed through on purpose: genolens-dd explains its refusals with
    the curated rationale, and replacing that with a generic message would throw away the only
    part of the response a user can act on.
    """
    if isinstance(exc, DrugDiscoveryRejected):
        return HTTPException(status_code=exc.status_code, detail=exc.detail)
    if isinstance(exc, DrugDiscoveryUnavailable):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    raise exc  # pragma: no cover - unknown type, do not swallow


@router.get("/indications")
async def list_drug_discovery_indications(
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Ce que l'UI a le droit de proposer : 33 indications, exclusions marquées, profils.

    Passe-plat strict — voir le docstring du module.
    """
    try:
        return await client.list_indications()
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)


@router.get("/status")
async def drug_discovery_status(
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Reachability and per-table readiness of genolens-dd.

    Reports `configured` so a missing key is distinguishable from an unreachable service —
    without it, both look like "Drug Discovery is down" and the wrong thing gets investigated.
    """
    if not client.is_configured:
        return {
            "configured": False,
            "reachable": None,
            "ready": None,
            "detail": "DD_API_KEY is not set on this server.",
        }
    try:
        readiness = await client.readyz()
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        return {"configured": True, "reachable": False, "ready": None, "detail": str(exc)}
    return {
        "configured": True,
        "reachable": True,
        "ready": readiness.get("ready"),
        "tables": readiness.get("tables"),
    }


@router.post("/runs", status_code=201)
async def create_drug_discovery_run(
    payload: RunRequest,
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Rank therapeutic targets for an indication."""
    try:
        return await client.create_run(
            indication=payload.indication,
            profile=payload.profile,
            allow_excluded=payload.allow_excluded,
        )
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)


@router.get("/runs/{run_id}")
async def get_drug_discovery_run(
    run_id: str,
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    try:
        return await client.get_run(run_id)
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)


@router.get("/runs/{run_id}/targets")
async def get_drug_discovery_targets(
    run_id: str,
    limit: int = Query(default=50, ge=1, le=1000),
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Ranked targets. `limit` is capped here as well as upstream.

    The cap is not defensive noise: genolens-dd can rank ~15,000 genes for one indication, and
    an uncapped default would ship all of them through two services to a browser.
    """
    try:
        return await client.get_targets(run_id, limit=limit)
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)


@router.get("/runs/{run_id}/report")
async def get_drug_discovery_report(
    run_id: str,
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    try:
        return await client.get_report(run_id)
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)


# ---------------------------------------------------------------------------
# Mode B — run a ranking against the user's own differential-expression comparison
# ---------------------------------------------------------------------------


class SignatureRunRequest(BaseModel):
    """Run a Drug Discovery ranking against a comparison the user owns."""

    dataset_id: UUID
    comparison_name: str

    #: Required, unlike mode A where null means pan-cancer. The mode B null draw is matched on
    #: tumour expression deciles, which only exist for a TCGA project — genolens-dd refuses a
    #: run without one rather than fall back to an unmatched draw labelled as matched.
    indication: str
    profile: str = "default_oncology"
    allow_excluded: bool = False

    padj_max: float = Field(default=0.05, gt=0, le=0.1)
    logfc_min: float = Field(default=1.0, ge=0, le=5)
    directions: Literal["both", "up", "down"] = "both"
    max_genes_per_condition: int = Field(default=1000, ge=1, le=2000)

    #: Replicate counts per condition name. **Required** — never inferred on this side. The
    #: upstream SIG001/SIG002 gate exists to catch underpowered arms, and a plausible default
    #: would walk straight past it.
    replicates: dict[str, int]
    allow_underpowered: bool = False
    #: Recorded and echoed back: without it, two runs give two p-values and "why this number?"
    #: has no answer six months later.
    seed: int = 1234


def _serialise_conditions(
    payload: SignaturePayload, *, include_genes: bool = False
) -> list[dict]:
    """`include_genes` only on the run response, never on the preview.

    The run response is the record of what was actually sent, and the frontend needs it to say
    which arm a hit came from — a Direction column derived from anything else would be a guess.
    The preview omits it because a list of a thousand symbols is noise before the run, and the
    caller already knows the counts.
    """
    return [
        {
            "name": condition.name,
            "direction": condition.direction,
            "n_genes": len(condition.genes),
            "n_available": condition.n_available,
            "truncated": condition.truncated,
            "replicates": condition.replicates,
            "replicates_source": condition.replicates_source,
            **({"genes": list(condition.genes)} if include_genes else {}),
        }
        for condition in payload.conditions
    ]


async def _load_owned_dataset(
    dataset_id: UUID, current_user: SupabaseUser, db: AsyncSession
) -> Dataset:
    """Dataset the caller may read, or 404.

    **Two gates, two different questions.** `require_team_plan` asks whether the caller may use
    the module at all; this asks whether the data is theirs. Neither substitutes for the other,
    and a plan check alone would let any TEAM user build a signature from another project's
    comparison.
    """
    from app.api.endpoints.datasets import _check_project_read_access

    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    await _check_project_read_access(dataset.project_id, current_user.user_id, db)
    return dataset


@router.get("/signature-preview")
async def preview_drug_discovery_signature(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    dataset_id: UUID = Query(...),
    comparison_name: str = Query(...),
    padj_max: float = Query(default=0.05, gt=0, le=0.1),
    logfc_min: float = Query(default=1.0, ge=0, le=5),
    directions: Literal["both", "up", "down"] = Query(default="both"),
    max_genes_per_condition: int = Query(default=1000, ge=1, le=2000),
    user: User = Depends(require_team_plan),
):
    """What would be sent, before anything is sent. **Makes no upstream call.**

    Exists so the user sees the gene counts, the resolved condition names and the replicate
    counts *before* their gene list leaves this service. A run that turns out to be
    misconfigured then costs a glance rather than a transmission.
    """
    dataset = await _load_owned_dataset(dataset_id, current_user, db)
    payload = await build_signature_payload(
        db, dataset, comparison_name,
        padj_max=padj_max, logfc_min=logfc_min, directions=directions,
        max_genes_per_condition=max_genes_per_condition,
    )
    return {
        "dataset_id": str(dataset_id),
        "comparison_name": comparison_name,
        "conditions": _serialise_conditions(payload),
        "needs_replicates": payload.needs_replicates,
        "species": payload.species,
        "warnings": list(payload.warnings),
    }


@router.post("/signature-runs", status_code=201)
async def create_drug_discovery_signature_run(
    payload: SignatureRunRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    limit: int = Query(default=100, ge=1, le=1000),
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Confront a comparison's DEG signature with the ranking for an indication.

    Creates the run and submits the signature **in one request, deliberately**. genolens-dd
    keeps runs in an in-memory dict; splitting the two across browser round-trips would let a
    redeploy evict the run in between, and the user would lose a signature they had already
    transmitted.

    Synchronous: the upstream permutation test is ~0.3 s at 1500 hits and our side is one
    indexed query, against a 30 s client timeout. The Celery job pattern used elsewhere exists
    for work that shells out to R for minutes.
    """
    dataset = await _load_owned_dataset(payload.dataset_id, current_user, db)
    signature = await build_signature_payload(
        db, dataset, payload.comparison_name,
        padj_max=payload.padj_max, logfc_min=payload.logfc_min,
        directions=payload.directions,
        max_genes_per_condition=payload.max_genes_per_condition,
        replicates_override=payload.replicates,
    )

    if not signature.conditions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No differentially expressed gene passes these thresholds in the requested "
                "direction(s). Loosen padj or |log2FC|."
            ),
        )
    if signature.needs_replicates:
        # Refused here rather than upstream: we know which condition is missing a count, and
        # sending a partial payload would surface as a rule-coded rejection about a name the
        # user never chose.
        missing = [c.name for c in signature.conditions if c.replicates is None]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Replicate count missing for condition(s): {', '.join(missing)}. "
                "Replicate counts are never guessed — provide them explicitly."
            ),
        )

    try:
        run = await client.create_run(
            indication=payload.indication,
            profile=payload.profile,
            allow_excluded=payload.allow_excluded,
        )
        result = await client.submit_signature(
            run["run_id"],
            client_id=str(dataset.project_id),
            genes_by_condition=signature.genes_by_condition(),
            replicates=signature.replicates(),
            seed=payload.seed,
            allow_underpowered=payload.allow_underpowered,
            limit=limit,
        )
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)

    return {
        "run_id": run["run_id"],
        "indication": payload.indication,
        "profile": payload.profile,
        "signature": {
            "conditions": _serialise_conditions(signature, include_genes=True),
            "genes_sent_total": signature.genes_sent_total,
            "warnings": list(signature.warnings),
        },
        "result": result,
    }


@router.get("/runs/{run_id}/signature/{signature_id}/report")
async def get_drug_discovery_signature_report(
    run_id: str,
    signature_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    user: User = Depends(require_team_plan),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    """Mode B report. A 422 here means the signature has no gene in the ranked universe."""
    try:
        return await client.get_signature_report(run_id, signature_id, limit=limit)
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)
