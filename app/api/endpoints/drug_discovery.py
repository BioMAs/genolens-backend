"""Drug Discovery endpoints — thin passthrough to the genolens-dd service.

**EVERY endpoint here requires an authenticated user, and that is the point of the module.**
genolens-dd is closed by an API key precisely so its rankings are not world-readable. An
unauthenticated route on this side would hand the same rankings to anyone who can reach
`api-v2.genolens.com`, re-opening the product through a different door while the key gave the
appearance of a closed one. That failure would look like a working feature, which is why it is
stated here rather than left to be noticed.

The handlers hold no business logic: genolens-dd owns the scoring contract, and a second
implementation of anything on this side is how the two services start disagreeing.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.supabase_auth import SupabaseUser
from app.services.drug_discovery import (
    DrugDiscoveryClient,
    DrugDiscoveryRejected,
    DrugDiscoveryUnavailable,
    get_drug_discovery_client,
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
    user: SupabaseUser = Depends(get_current_user),
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
    user: SupabaseUser = Depends(get_current_user),
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
    user: SupabaseUser = Depends(get_current_user),
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
    user: SupabaseUser = Depends(get_current_user),
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
    user: SupabaseUser = Depends(get_current_user),
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
    user: SupabaseUser = Depends(get_current_user),
    client: DrugDiscoveryClient = Depends(get_drug_discovery_client),
):
    try:
        return await client.get_report(run_id)
    except (DrugDiscoveryRejected, DrugDiscoveryUnavailable) as exc:
        raise _reraise(exc)
