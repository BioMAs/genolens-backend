"""
Pricing grid endpoint — public, no authentication required.

The grid is a commercial promise: it must be readable by the pricing page and
the marketing site before anyone signs in. Serving it from here is what makes
`app/config/pricing.json` the single source of truth instead of one more copy
among the four hard-coded tables the frontend used to carry.
"""
import hashlib
import json

from fastapi import APIRouter, Request, Response, status

from app.core.pricing import get_pricing

router = APIRouter(prefix="/pricing", tags=["pricing"])

# The grid ships with the image and only changes on deploy, so it can be cached
# hard. `must-revalidate` keeps a price correction from lingering behind a stale
# CDN entry once the ETag changes.
_CACHE_CONTROL = "public, max-age=300, must-revalidate"


# HEAD as well as GET: this is a public, cacheable resource, and proxies and
# uptime checks issue HEAD against it. Without this they get a 405.
@router.api_route("", methods=["GET", "HEAD"])
@router.api_route("/", methods=["GET", "HEAD"])
def get_pricing_grid(request: Request, response: Response):
    """Return the active pricing grid: plans, add-on modules, quotas, rights."""
    cfg = get_pricing()
    # by_alias so `$comment` round-trips; exclude_none keeps the payload to what
    # the grid actually states, which is what lets a consumer tell "not stated"
    # from "stated false".
    payload = cfg.model_dump(mode="json", by_alias=True, exclude_none=True)

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    etag = f'W/"{hashlib.sha256(body.encode()).hexdigest()[:32]}"'

    if request.headers.get("if-none-match") == etag:
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED,
            headers={"ETag": etag, "Cache-Control": _CACHE_CONTROL},
        )

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = _CACHE_CONTROL
    return payload
