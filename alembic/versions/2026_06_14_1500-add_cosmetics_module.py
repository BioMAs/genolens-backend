"""add_cosmetics_module

Adds the Cosmetics add-on module:
  - users.cosmetics_module_enabled (per-user unlock toggled by admins)
  - claim_pathway_mappings (curated pathway -> cosmetic claim referential)
  - claim_references (category-level citations)
  - cosmetic_claims (canonical claim taxonomy)

Idempotent so it is safe to re-run on databases that partially migrated.

Revision ID: cosmetics_module_001
Revises: report_jobs_analysis_001
Create Date: 2026-06-14
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "cosmetics_module_001"
down_revision: Union[str, None] = "report_jobs_analysis_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # --- users.cosmetics_module_enabled ----------------------------------
    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "cosmetics_module_enabled" not in user_columns:
        op.add_column(
            "users",
            sa.Column(
                "cosmetics_module_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    # --- enums -----------------------------------------------------------
    claim_direction = postgresql.ENUM(
        "UP", "DOWN", "BOTH", "UNKNOWN", "AVOID", name="claim_direction_enum"
    )
    evidence_level = postgresql.ENUM(
        "HIGH", "MODERATE", "LOW", name="evidence_level_enum"
    )
    claim_direction.create(bind, checkfirst=True)
    evidence_level.create(bind, checkfirst=True)

    existing_tables = set(inspector.get_table_names())

    # --- claim_pathway_mappings -----------------------------------------
    if "claim_pathway_mappings" not in existing_tables:
        op.create_table(
            "claim_pathway_mappings",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("term_id", sa.String(length=64), nullable=False),
            sa.Column("term_id_normalized", sa.String(length=64), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("common_abbrev", sa.String(length=255), nullable=True),
            sa.Column("suggested_acronym", sa.String(length=64), nullable=True),
            sa.Column("original_claims", sa.Text(), nullable=True),
            sa.Column("original_direction", sa.String(length=32), nullable=True),
            sa.Column("updated_claim_framing", sa.Text(), nullable=True),
            sa.Column(
                "updated_direction",
                postgresql.ENUM(name="claim_direction_enum", create_type=False),
                nullable=False,
            ),
            sa.Column("category", sa.String(length=64), nullable=True),
            sa.Column(
                "evidence_level",
                postgresql.ENUM(name="evidence_level_enum", create_type=False),
                nullable=False,
            ),
            sa.Column("rationale", sa.Text(), nullable=True),
            sa.Column("caveats", sa.Text(), nullable=True),
            sa.Column("ref_cat", sa.String(length=64), nullable=True),
            sa.Column("canonical_claims", sa.JSON(), nullable=False),
            sa.Column(
                "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("term_id"),
        )
        op.create_index(
            "ix_claim_pathway_mappings_term_id", "claim_pathway_mappings", ["term_id"]
        )
        op.create_index(
            "ix_claim_pathway_mappings_term_id_normalized",
            "claim_pathway_mappings",
            ["term_id_normalized"],
        )
        op.create_index(
            "ix_claim_pathway_mappings_category", "claim_pathway_mappings", ["category"]
        )
        op.create_index(
            "ix_claim_pathway_mappings_norm_active",
            "claim_pathway_mappings",
            ["term_id_normalized", "is_active"],
        )

    # --- claim_references ------------------------------------------------
    if "claim_references" not in existing_tables:
        op.create_table(
            "claim_references",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("ref_cat", sa.String(length=64), nullable=False),
            sa.Column("source_summary", sa.Text(), nullable=True),
            sa.Column("url", sa.String(length=1000), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_claim_references_ref_cat", "claim_references", ["ref_cat"])

    # --- cosmetic_claims -------------------------------------------------
    if "cosmetic_claims" not in existing_tables:
        op.create_table(
            "cosmetic_claims",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("slug", sa.String(length=64), nullable=False),
            sa.Column("label", sa.String(length=128), nullable=False),
            sa.Column("skin_zone", sa.String(length=64), nullable=True),
            sa.Column("color", sa.String(length=16), nullable=True),
            sa.Column("icon", sa.String(length=64), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slug"),
        )
        op.create_index("ix_cosmetic_claims_slug", "cosmetic_claims", ["slug"])

    # --- cosmetic_interpretations (AI cache) -----------------------------
    if "cosmetic_interpretations" not in existing_tables:
        op.create_table(
            "cosmetic_interpretations",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("dataset_id", sa.Uuid(), nullable=False),
            sa.Column("comparison_name", sa.String(length=255), nullable=False),
            sa.Column("interpretation", sa.Text(), nullable=False),
            sa.Column("model", sa.String(length=100), nullable=False),
            sa.Column("claims_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_cosmetic_interpretations_dataset_id",
            "cosmetic_interpretations",
            ["dataset_id"],
        )
        op.create_index(
            "ix_cosmetic_interpretations_comparison_name",
            "cosmetic_interpretations",
            ["comparison_name"],
        )
        op.create_index(
            "ix_cosmetic_interpretations_dataset_comparison",
            "cosmetic_interpretations",
            ["dataset_id", "comparison_name"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "cosmetic_interpretations" in existing_tables:
        for idx in (
            "ix_cosmetic_interpretations_dataset_comparison",
            "ix_cosmetic_interpretations_comparison_name",
            "ix_cosmetic_interpretations_dataset_id",
        ):
            op.drop_index(idx, table_name="cosmetic_interpretations")
        op.drop_table("cosmetic_interpretations")

    if "cosmetic_claims" in existing_tables:
        op.drop_index("ix_cosmetic_claims_slug", table_name="cosmetic_claims")
        op.drop_table("cosmetic_claims")

    if "claim_references" in existing_tables:
        op.drop_index("ix_claim_references_ref_cat", table_name="claim_references")
        op.drop_table("claim_references")

    if "claim_pathway_mappings" in existing_tables:
        for idx in (
            "ix_claim_pathway_mappings_norm_active",
            "ix_claim_pathway_mappings_category",
            "ix_claim_pathway_mappings_term_id_normalized",
            "ix_claim_pathway_mappings_term_id",
        ):
            op.drop_index(idx, table_name="claim_pathway_mappings")
        op.drop_table("claim_pathway_mappings")

    op.execute("DROP TYPE IF EXISTS claim_direction_enum")
    op.execute("DROP TYPE IF EXISTS evidence_level_enum")

    user_columns = {c["name"] for c in inspector.get_columns("users")}
    if "cosmetics_module_enabled" in user_columns:
        op.drop_column("users", "cosmetics_module_enabled")
