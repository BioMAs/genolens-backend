"""AI texts tied to their inputs: cosmetic_interpretations.inputs_sha256, report_jobs.ai_texts

cosmetic_interpretations.inputs_sha256: the fingerprint of the nova a cached AI reading was written
from (claims, unlinked sub-groups, alpha, referential version, enrichment datasets). A text whose
fingerprint is not the current nova's is stale: it is neither served as current nor printed. The
text's own model is hashed on both sides (the row's, not the configured one), so a change of model
never makes a text stale. Rows written before this revision have no fingerprint (NULL) and so count
as stale.
report_jobs.ai_texts: the AI texts a report really printed, [{kind, model, generated_at}]
(NULL: not recorded, [] : none printed).
Both columns are nullable. Idempotent so it is safe to re-run on partially migrated databases.

Revision ID: nova_honesty_001
Revises: service_tokens_001
Create Date: 2026-11-05 10:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "nova_honesty_001"
down_revision: Union[str, None] = "service_tokens_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if "inputs_sha256" not in _columns("cosmetic_interpretations"):
        op.add_column(
            "cosmetic_interpretations",
            sa.Column("inputs_sha256", sa.String(length=64), nullable=True),
        )
    if "ai_texts" not in _columns("report_jobs"):
        op.add_column("report_jobs", sa.Column("ai_texts", sa.JSON(), nullable=True))


def downgrade() -> None:
    if "ai_texts" in _columns("report_jobs"):
        op.drop_column("report_jobs", "ai_texts")
    if "inputs_sha256" in _columns("cosmetic_interpretations"):
        op.drop_column("cosmetic_interpretations", "inputs_sha256")
