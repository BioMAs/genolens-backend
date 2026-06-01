"""merge_heads

Merge the main branch (2026_05_06_0001) with the license/business-model branch
(report_jobs_001) into a single head.

Revision ID: merge_heads_001
Revises: 2026_05_06_0001, report_jobs_001
Create Date: 2026-06-01
"""
from typing import Sequence, Union

revision: str = "merge_heads_001"
down_revision: Union[str, None] = ("2026_05_06_0001", "report_jobs_001")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
