"""
OmicSkin score — one 0–100 figure per cosmetic claim, read off the CosmOmics Pathway Nova.

    score = round(100 × coherence × min(1, best_nl / 8))

  coherence  share of the claim's evidence that goes in the expected direction,
             S_pro / (S_pro + S_anti) (nova payload `coherence`, 0..1)
  best_nl    −log10 of the Stouffer p of the claim's most significant sub-group (nova `best_nl`);
             the factor saturates once that sub-group reaches p = 10⁻⁸.

Bands: ≥ 75 strong support · 50–74 moderate · 25–49 weak · < 25 not supported. A claim with no
term at adjusted p < alpha is absent from the nova payload (`build_nova` drops it) and receives
no score: "no significant evidence".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.services.cosmomics_nova import CLAIM_NAMES

SATURATION_NL = 8.0

Band = Literal["strong", "moderate", "weak", "not_supported", "no_evidence"]

BAND_LABELS: dict[str, str] = {
    "strong": "Strong support",
    "moderate": "Moderate support",
    "weak": "Weak support",
    "not_supported": "Not supported",
    "no_evidence": "No significant evidence",
}

# All nine claims of the referential, in the order the nova names them.
ALL_CLAIMS: tuple[str, ...] = tuple(CLAIM_NAMES.values())


def omicskin_score(coherence: float, best_nl: float) -> int:
    """0–100 score of one claim. Same arithmetic (Python `round`) as the report reference."""
    coherence = min(max(float(coherence), 0.0), 1.0)
    significance = min(1.0, max(float(best_nl), 0.0) / SATURATION_NL)
    return int(round(100 * coherence * significance))


def omicskin_band(score: Optional[int]) -> Band:
    if score is None:
        return "no_evidence"
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    if score >= 25:
        return "weak"
    return "not_supported"


@dataclass(frozen=True)
class ClaimScore:
    name: str
    score: Optional[int]
    band: Band
    claim: Optional[dict[str, Any]]  # the nova payload entry, None without evidence

    @property
    def label(self) -> str:
        return BAND_LABELS[self.band]

    @property
    def scored(self) -> bool:
        return self.score is not None


def score_claims(nova: dict[str, Any]) -> list[ClaimScore]:
    """Scores of the nine claims: scored ones first (highest score first), then the others.

    Claims present in the nova payload carry at least one term at adjusted p < alpha; the
    remaining claims of the referential have no significant evidence and no score.
    """
    scored: list[ClaimScore] = []
    seen: set[str] = set()
    for claim in nova.get("claims") or []:
        name = str(claim["name"])
        seen.add(name)
        value = omicskin_score(claim.get("coherence") or 0.0, claim.get("best_nl") or 0.0)
        scored.append(ClaimScore(name, value, omicskin_band(value), claim))
    order = {name: i for i, name in enumerate(ALL_CLAIMS)}
    scored.sort(key=lambda c: (-(c.score or 0), order.get(c.name, len(order)), c.name))
    unscored = [
        ClaimScore(name, None, "no_evidence", None) for name in ALL_CLAIMS if name not in seen
    ]
    return scored + unscored
