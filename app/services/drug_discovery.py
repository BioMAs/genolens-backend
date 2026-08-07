"""Client for the genolens-dd service (pan-cancer therapeutic target prioritisation).

`genolens-dd` is a separate service with its own analytical stack (DuckDB/Polars, Python 3.12)
and its own confidential client data. The backend therefore talks to it over HTTP, the same way
it would any third party — see the module docstring of `external_integrations.py` for the same
httpx/timeout shape.

Two things in here are security-relevant rather than cosmetic.

**The API key never leaves this process.** genolens-dd stores only the SHA-256 of the key, so
this backend is the single place the plaintext exists. It goes into a request header and nowhere
else: not into a log line, not into an error message, not into a response body. `__repr__` is
overridden for the same reason — a dataclass repr in a traceback is a classic way for a secret
to reach a log aggregator.

**An upstream 401 is OUR fault, not the caller's.** Mapping it straight through would tell a
logged-in user "unauthorised" when the truth is that this backend's key is wrong or revoked.
The user would retry, re-authenticate, and file a support ticket about the wrong system. It maps
to 502 and logs loudly instead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

#: Header expected by genolens-dd. Deliberately NOT `Authorization`, which Traefik's basicauth
#: middleware consumes on that host — two layers on one header produce a failure whose cause is
#: not readable from either codebase.
API_KEY_HEADER = "X-API-Key"


class DrugDiscoveryUnavailable(RuntimeError):
    """The service is unreachable, misconfigured, or failed. Maps to 502/503/504."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class DrugDiscoveryRejected(ValueError):
    """The service refused the request on its own terms (4xx other than 401).

    Carries the upstream status and detail through, because genolens-dd's refusals are
    *informative* — an excluded indication comes back with the curated rationale explaining
    why, and swallowing that would turn a well-explained 422 into a blank 502.
    """

    def __init__(self, message: str, status_code: int, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


@dataclass
class DrugDiscoveryClient:
    """Thin async client. One method per genolens-dd endpoint, no business logic."""

    base_url: str
    api_key: str
    timeout: float = 30.0
    #: Injected by tests as `httpx.MockTransport`, so the suite makes no network call.
    transport: Optional[httpx.AsyncBaseTransport] = None

    def __repr__(self) -> str:  # pragma: no cover - trivial, but see module docstring
        """Never render the key. A dataclass repr in a traceback reaches log aggregators."""
        configured = "set" if self.api_key else "unset"
        return f"DrugDiscoveryClient(base_url={self.base_url!r}, api_key=<{configured}>)"

    @classmethod
    def from_settings(cls) -> "DrugDiscoveryClient":
        return cls(
            base_url=settings.DD_BASE_URL.rstrip("/"),
            api_key=settings.DD_API_KEY,
            timeout=settings.DD_TIMEOUT_SECONDS,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
        authenticated: bool = True,
        preserve_503: bool = False,
    ) -> Any:
        """Single exit point to genolens-dd, so the key is attached in exactly one place."""
        if authenticated and not self.is_configured:
            raise DrugDiscoveryUnavailable(
                "Drug Discovery is not configured on this server (DD_API_KEY is empty).",
                status_code=503,
            )

        headers = {API_KEY_HEADER: self.api_key} if authenticated else {}
        url = f"{self.base_url}{path}"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.request(
                    method, url, json=json_body, params=params, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise DrugDiscoveryUnavailable(
                f"Drug Discovery timed out after {self.timeout:.0f}s.", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            # `str(exc)` on an httpx error contains the URL, never the headers — safe to log.
            logger.warning("genolens-dd unreachable: %s", exc)
            raise DrugDiscoveryUnavailable(
                "Drug Discovery is unreachable.", status_code=502
            ) from exc

        return self._interpret(response, preserve_503=preserve_503)

    def _interpret(self, response: httpx.Response, *, preserve_503: bool = False) -> Any:
        """Map an upstream status onto ours. The 401 branch is the one that matters."""
        if response.status_code == 401:
            # OUR key is wrong or revoked. Surfacing 401 would blame the logged-in user.
            logger.error(
                "genolens-dd rejected our API key (401). Check DD_API_KEY against the "
                "DD_API_KEYS fingerprint on the genolens-dd host."
            )
            raise DrugDiscoveryUnavailable(
                "Drug Discovery credentials are invalid on this server.", status_code=502
            )

        if 400 <= response.status_code < 500:
            raise DrugDiscoveryRejected(
                f"Drug Discovery refused the request ({response.status_code}).",
                status_code=response.status_code,
                detail=self._detail(response),
            )

        # A 503 from /readyz is genolens-dd's normal answer when the reference socle is
        # incomplete — its body names exactly which tables are missing via `tables`. That is
        # a business response, not an outage, and mapping it onto DrugDiscoveryUnavailable like
        # any other 5xx would throw the body away: /status would then report `reachable: false`
        # and a caller would go looking for a network problem instead of a data problem. Only
        # `readyz()` opts into this by passing `preserve_503=True` — every other 5xx, on every
        # other endpoint, still becomes 502 below, unchanged.
        if preserve_503 and response.status_code == 503:
            try:
                return response.json()
            except ValueError as exc:
                raise DrugDiscoveryUnavailable(
                    "Drug Discovery returned a malformed response.", status_code=502
                ) from exc

        if response.status_code >= 500:
            logger.warning("genolens-dd returned %s", response.status_code)
            raise DrugDiscoveryUnavailable(
                "Drug Discovery failed to process the request.", status_code=502
            )

        try:
            return response.json()
        except ValueError as exc:
            raise DrugDiscoveryUnavailable(
                "Drug Discovery returned a malformed response.", status_code=502
            ) from exc

    @staticmethod
    def _detail(response: httpx.Response) -> Any:
        """Upstream `detail` when there is one, so a 422 keeps its curated rationale."""
        try:
            body = response.json()
        except ValueError:
            return response.text[:500]
        return body.get("detail", body) if isinstance(body, dict) else body

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """Liveness. Unauthenticated upstream, so it works even with no key configured —
        which is what makes it useful for diagnosing a missing key."""
        return await self._request("GET", "/health", authenticated=False)

    async def readyz(self) -> dict:
        """Per-table readiness. Unauthenticated upstream by design (see genolens-dd).

        `preserve_503=True`: genolens-dd answers 503 here as its nominal signal that the
        reference socle is incomplete, with a body (`tables`) saying exactly what to rebuild.
        Losing that body on a legitimate deploy-day state would leave `/status` reporting
        "unreachable" when the service is up and simply missing data.
        """
        return await self._request("GET", "/readyz", authenticated=False, preserve_503=True)

    async def list_indications(self) -> dict:
        """Catalogue des indications et des profils servables.

        Authentifié : genolens-dd garde cette route, et la liste des indications couvertes
        renseigne sur le socle de données.
        """
        return await self._request("GET", "/indications")

    async def create_run(
        self,
        *,
        indication: Optional[str] = None,
        profile: str = "default_oncology",
        allow_excluded: bool = False,
    ) -> dict:
        body: dict[str, Any] = {"indication": indication, "profile": profile}
        if allow_excluded:
            body["allow_excluded"] = True
        return await self._request("POST", "/runs", json_body=body)

    async def get_run(self, run_id: str) -> dict:
        return await self._request("GET", f"/runs/{run_id}")

    async def get_targets(self, run_id: str, *, limit: int = 50) -> dict:
        return await self._request("GET", f"/runs/{run_id}/targets", params={"limit": limit})

    async def get_report(self, run_id: str) -> dict:
        return await self._request("GET", f"/runs/{run_id}/report")


def get_drug_discovery_client() -> DrugDiscoveryClient:
    """FastAPI dependency. Built per request; httpx clients are created per call anyway."""
    return DrugDiscoveryClient.from_settings()
