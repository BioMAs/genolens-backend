"""
Pricing configuration — the single source of truth for plans, quotas and
entitlements.

Everything commercial lives in `app/config/pricing.json` and is served to every
consumer (this app, the pricing page, the marketing site) through
`GET /api/v1/pricing`. No price, quota or entitlement may be hard-coded
anywhere else.

Two files, one schema:

* `pricing.json` — the **active** grid. It currently describes the three plans
  exactly as `User`'s properties enforce them today, so introducing this module
  changes no behaviour. `tests/test_pricing_config.py` fails if the two ever
  diverge.
* `pricing.2026-09-draft.json` — the proposed grid (annual quota of interpreted
  contrasts, five plans). Validated by this same schema in CI without being
  served, so the day it is activated we already know it parses and its
  entitlement keys are complete.

This is the `app/core/plan_config.py` that `docs/features/account-management.md`
has referenced for months without it ever existing.

Limit conventions
-----------------
A limit is `int | "unlimited" | "custom" | None`. `resolve_limit()` normalises
it to `int | None`, where **None means unlimited** — the same convention the
`User` properties already use (`comparisons_quota` returns None for
ON_PREMISE). `0` means "none included", which is not the same thing: the Access
plan of the draft grid bundles zero contrasts and sells them a la carte.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# app/config/ holds shipped, version-controlled configuration. Deliberately NOT
# app/data/, which .gitignore excludes for runtime data artifacts (anno_db and
# friends) — a pricing grid left there would never be committed and would go
# missing on deploy.
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ACTIVE_GRID = _CONFIG_DIR / "pricing.json"
DRAFT_GRID = _CONFIG_DIR / "pricing.2026-09-draft.json"

#: A quota or cap as written in JSON. See "Limit conventions" above.
Limit = Union[int, Literal["unlimited", "custom"], None]

#: How AI interpretation is dispensed. Deliberately three-valued, not a bool:
#: the Access plan grants access but bills every act, which `can_use_ai` (a
#: bool) cannot express.
AiInterpretationMode = Literal["none", "metered_a_la_carte", "quota"]

#: Entitlement keys owned by an add-on module rather than by a plan. Listing
#: them on a plan is rejected: it would create two contradictory sources of
#: truth for the same capability (a lens_12 customer would get GSEA from their
#: plan while `require_scientific_access` still demands the module).
ADDON_OWNED_ENTITLEMENTS = frozenset(
    {"gsea", "gsea_leading_edge", "custom_gene_sets", "signature_score",
     "contrast_scatter", "deg_patterns", "report_customization",
     "cosmetic_claims", "drug_discovery"}
)


def resolve_limit(value: Limit) -> Optional[int]:
    """Normalise a JSON limit to `int | None`, where None means unlimited.

    "custom" (negotiated per deal, e.g. Enterprise) also resolves to None: an
    unnegotiated Enterprise account must not be capped by a placeholder.
    """
    if value is None or value in ("unlimited", "custom"):
        return None
    return int(value)


class _Strict(BaseModel):
    """Reject unknown keys everywhere.

    This is the whole point of the schema: a typo in an entitlement name must
    break CI, not silently grant or deny a capability at runtime.
    """
    model_config = ConfigDict(extra="forbid")


class Entitlements(_Strict):
    """Capability flags. `None` means "not stated at this level"."""

    differential_expression: Optional[bool] = None
    clustering_basic: Optional[bool] = None
    clustering_full: Optional[bool] = None
    enrichment_go_bp: Optional[bool] = None
    enrichment_go_kegg_reactome: Optional[bool] = None
    plot_export: Optional[bool] = None
    ai_interpretation: Optional[AiInterpretationMode] = None
    knowledge_graph: Optional[Union[bool, AiInterpretationMode]] = None
    cross_project_comparison: Optional[bool] = None
    rest_api: Optional[bool] = None
    api_rate_limit_per_min: Optional[int] = None
    sso_saml: Optional[bool] = None
    on_premise: Optional[bool] = None
    support: Optional[str] = None
    multi_comparison: Optional[bool] = None
    advanced_export: Optional[bool] = None

    # Add-on owned (valid on an Addon, rejected on a Plan — see Plan validator)
    gsea: Optional[bool] = None
    gsea_leading_edge: Optional[bool] = None
    custom_gene_sets: Optional[bool] = None
    signature_score: Optional[bool] = None
    contrast_scatter: Optional[bool] = None
    deg_patterns: Optional[bool] = None
    report_customization: Optional[bool] = None
    cosmetic_claims: Optional[bool] = None
    drug_discovery: Optional[bool] = None

    def stated(self) -> dict:
        """Only the keys actually present in the JSON."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BillableUnit(_Strict):
    id: str
    label_fr: str
    label_en: str
    definition: str
    identity_hash: Optional[str] = None
    identity_excludes: list[str] = Field(default_factory=list)
    not_billable: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None


class Economics(BaseModel):
    """Cost model. Free-form on purpose — commercial arithmetic, not runtime
    behaviour. Kept loose so the draft's notes survive a round trip."""
    model_config = ConfigDict(extra="allow")

    cost_per_interpreted_contrast_automated: Optional[float] = None
    cost_per_interpreted_contrast_manual: Optional[float] = None
    margin_floor_pct: Optional[float] = None


class ALaCarte(BaseModel):
    model_config = ConfigDict(extra="allow")
    interpreted_contrast: Optional[float] = None


class MarketingFeature(_Strict):
    """One bullet on the public pricing card.

    Copy lives in the grid because the grid is the source of truth for the
    marketing site as well as the app. Careful: a bullet is a commercial
    promise, NOT an entitlement — several of the current ones are enforced by
    nothing (see each plan's `note`). Read `entitlements` to decide access;
    read these only to render the card.
    """
    label: str
    included: bool = True


class Plan(_Strict):
    id: str
    name_fr: str
    name_en: str
    order: int
    most_popular: bool = False
    replaces: Optional[str] = None

    price_monthly: Optional[float] = None
    price_annual: Optional[float] = None
    price_monthly_equivalent: Optional[float] = None
    price_annual_breakdown: Optional[str] = None
    pricing_display: Optional[str] = None
    pricing_display_en: Optional[str] = None
    billing: list[Literal["monthly", "annual", "custom"]] = Field(default_factory=list)
    commitment_months: int = 0

    # The billable axis. `quota_period` is explicit so the move from the
    # current monthly counter to the draft's annual quota is a visible diff.
    contrast_quota: Limit = None
    quota_period: Literal["monthly", "annual"] = "annual"
    contrast_unit_price: Optional[float] = None
    contrast_unit_price_floor: Optional[float] = None
    contrast_unit_price_floor_note: Optional[str] = None
    discount_vs_a_la_carte_pct: Optional[float] = None
    contrast_margin_pct: Optional[float] = None

    seats: Limit = None
    max_projects: Limit = None
    #: Account-wide dataset cap (the draft grid's axis).
    datasets_limit: Limit = None
    #: Per-project dataset cap (what the code enforces today).
    datasets_limit_per_project: Limit = None
    storage_mb_per_dataset: Limit = None
    raw_data_retention_months: Limit = None
    delivered_dossier_retention: Optional[str] = None

    entitlements: Entitlements = Field(default_factory=Entitlements)

    # Public card copy (see MarketingFeature)
    description_en: Optional[str] = None
    engagement_en: Optional[str] = None
    cta_label_en: Optional[str] = None
    marketing_features: list[MarketingFeature] = Field(default_factory=list)

    on_premise_licence_annual: Optional[float] = None
    beta_testing_contract_annual: Optional[float] = None
    positioning_fr: Optional[str] = None
    note: Optional[str] = None

    @model_validator(mode="after")
    def _reject_addon_owned_entitlements(self) -> "Plan":
        stated = set(self.entitlements.stated())
        clash = sorted(stated & ADDON_OWNED_ENTITLEMENTS)
        if clash:
            raise ValueError(
                f"plan {self.id!r} states add-on owned entitlements {clash}; "
                "these belong under `addons`, otherwise a plan flag and a "
                "`*_module_enabled` guard would disagree at runtime"
            )
        return self

    # Resolved limits — None means unlimited. Use these, never the raw fields.
    @property
    def resolved_contrast_quota(self) -> Optional[int]:
        return resolve_limit(self.contrast_quota)

    @property
    def resolved_seats(self) -> Optional[int]:
        return resolve_limit(self.seats)

    @property
    def resolved_max_projects(self) -> Optional[int]:
        return resolve_limit(self.max_projects)

    @property
    def resolved_datasets_limit(self) -> Optional[int]:
        return resolve_limit(self.datasets_limit)

    @property
    def resolved_datasets_limit_per_project(self) -> Optional[int]:
        return resolve_limit(self.datasets_limit_per_project)


class Addon(_Strict):
    """A module sold independently of the plan.

    `module_flag` names the existing `User` boolean column, which is what keeps
    this config backward-compatible: the guards in
    `app/api/deps/subscription.py` keep reading the column, the config only
    describes and prices it.
    """
    id: str
    name_fr: str
    name_en: str
    order: int
    module_flag: str
    entitlements: Entitlements = Field(default_factory=Entitlements)
    price_monthly: Optional[float] = None
    price_annual: Optional[float] = None
    pricing_display: Optional[str] = None
    billing: list[Literal["monthly", "annual", "custom"]] = Field(default_factory=list)
    positioning_fr: Optional[str] = None
    note: Optional[str] = None

    @field_validator("module_flag")
    @classmethod
    def _looks_like_a_column(cls, v: str) -> str:
        if not v.endswith("_module_enabled"):
            raise ValueError(
                f"module_flag {v!r} must name a User boolean column "
                "(*_module_enabled)"
            )
        return v


class Overage(_Strict):
    policy: Literal["allow_and_bill", "block"] = "block"
    unit_price_rule: Optional[str] = None
    note: Optional[str] = None
    soft_warning_at_pct: Optional[int] = None
    hard_block: bool = True
    invoice_cadence: Optional[str] = None


class Rollover(_Strict):
    enabled: bool = False
    rationale: Optional[str] = None


class DataRights(BaseModel):
    model_config = ConfigDict(extra="allow")
    delivered_dossier: Optional[str] = None
    raw_data: Optional[str] = None
    on_subscription_end: Optional[str] = None


class PricingConfig(BaseModel):
    """The whole grid. Extra top-level keys are allowed so commercial annexes
    (retainer mapping, migration rules, known inconsistencies) round-trip
    without this schema having to model decisions that never reach the code."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str
    status: str
    currency: str = "EUR"
    tax_mode: str = "HT"
    comment: Optional[str] = Field(default=None, alias="$comment")
    billable_unit: Optional[BillableUnit] = None
    economics: Optional[Economics] = None
    a_la_carte: Optional[ALaCarte] = None
    plans: list[Plan]
    addons: list[Addon] = Field(default_factory=list)
    overage: Optional[Overage] = None
    rollover: Optional[Rollover] = None
    data_rights: Optional[DataRights] = None

    @field_validator("plans")
    @classmethod
    def _plans_unique_and_non_empty(cls, v: list[Plan]) -> list[Plan]:
        if not v:
            raise ValueError("pricing config must define at least one plan")
        ids = [p.id for p in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate plan ids: {dupes}")
        return v

    @field_validator("addons")
    @classmethod
    def _addons_unique(cls, v: list[Addon]) -> list[Addon]:
        for field in ("id", "module_flag"):
            vals = [getattr(a, field) for a in v]
            dupes = sorted({x for x in vals if vals.count(x) > 1})
            if dupes:
                raise ValueError(f"duplicate addon {field}: {dupes}")
        return v

    def get_plan(self, plan_id: str) -> Plan:
        """Look a plan up by id (accepts the `SubscriptionPlan` enum value)."""
        key = str(getattr(plan_id, "value", plan_id))
        for p in self.plans:
            if p.id == key:
                return p
        raise KeyError(
            f"no plan {key!r} in pricing grid {self.version}; "
            f"known: {[p.id for p in self.plans]}"
        )

    def get_addon(self, addon_id: str) -> Addon:
        for a in self.addons:
            if a.id == addon_id:
                return a
        raise KeyError(
            f"no addon {addon_id!r} in pricing grid {self.version}; "
            f"known: {[a.id for a in self.addons]}"
        )

    @property
    def plans_ordered(self) -> list[Plan]:
        return sorted(self.plans, key=lambda p: p.order)

    @property
    def addons_ordered(self) -> list[Addon]:
        return sorted(self.addons, key=lambda a: a.order)


def load_pricing(path: Path) -> PricingConfig:
    """Parse and validate a grid file. Raises on any schema violation."""
    return PricingConfig.model_validate(json.loads(path.read_text()))


@lru_cache(maxsize=1)
def get_pricing() -> PricingConfig:
    """The active grid, parsed once. Call `.cache_clear()` in tests."""
    return load_pricing(ACTIVE_GRID)
