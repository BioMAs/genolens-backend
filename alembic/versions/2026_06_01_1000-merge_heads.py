"""merge_heads

Point this no-op migration at the current single chain head (report_jobs_001).
The previously referenced revision '2026_05_06_0001' does not exist.

Revision ID: merge_heads_001
Revises: report_jobs_001
Create Date: 2026-06-01
"""
from typing import Sequence, Union

revision: str = "merge_heads_001"
down_revision: Union[str, None] = "report_jobs_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
