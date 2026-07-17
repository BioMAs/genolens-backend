"""scope gene_sets for custom (project-scoped) sets

Adds nullable project_id + user_id to gene_sets and replaces the global unique
index (name, database) with two partial unique indexes: built-in sets stay
globally unique, custom sets are unique per project.

Revision ID: scope_custom_gene_sets_001
Revises: gsea_jobs_001
Create Date: 2026-07-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "scope_custom_gene_sets_001"
down_revision: Union[str, None] = "gsea_jobs_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("gene_sets", sa.Column("project_id", sa.UUID(), nullable=True))
    op.add_column("gene_sets", sa.Column("user_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "fk_gene_sets_project_id", "gene_sets", "projects",
        ["project_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_gene_sets_project_id", "gene_sets", ["project_id"])

    # Replace the global unique index with two partial unique indexes.
    op.drop_index("ix_gene_sets_name_database", table_name="gene_sets")
    op.create_index(
        "ix_gene_sets_name_db_builtin", "gene_sets", ["name", "database"],
        unique=True, postgresql_where=sa.text("project_id IS NULL"),
    )
    op.create_index(
        "ix_gene_sets_name_db_project", "gene_sets", ["name", "database", "project_id"],
        unique=True, postgresql_where=sa.text("project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_gene_sets_name_db_project", table_name="gene_sets")
    op.drop_index("ix_gene_sets_name_db_builtin", table_name="gene_sets")
    op.create_index("ix_gene_sets_name_database", "gene_sets", ["name", "database"], unique=True)
    op.drop_index("ix_gene_sets_project_id", table_name="gene_sets")
    op.drop_constraint("fk_gene_sets_project_id", "gene_sets", type_="foreignkey")
    op.drop_column("gene_sets", "user_id")
    op.drop_column("gene_sets", "project_id")
