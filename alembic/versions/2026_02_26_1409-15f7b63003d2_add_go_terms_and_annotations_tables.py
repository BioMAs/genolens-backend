"""add_go_terms_and_annotations_tables

Revision ID: 15f7b63003d2
Revises: a0de1acb7e87
Create Date: 2026-02-26 14:09:06.999534

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '15f7b63003d2'
down_revision: Union[str, None] = 'a0de1acb7e87'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pg_trgm extension for fuzzy text search (if not already enabled)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    # Create go_terms table
    op.create_table(
        'go_terms',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('go_id', sa.String(20), nullable=False, unique=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('namespace', sa.String(50), nullable=False,
                  comment='biological_process, molecular_function, or cellular_component'),
        sa.Column('definition', sa.Text, nullable=True),
        sa.Column('is_a', postgresql.JSON, nullable=False, server_default='[]',
                  comment='Parent GO IDs (is_a relationships)'),
        sa.Column('part_of', postgresql.JSON, nullable=False, server_default='[]',
                  comment='Parent GO IDs (part_of relationships)'),
        sa.Column('regulates', postgresql.JSON, nullable=False, server_default='[]',
                  comment='GO IDs this term regulates'),
        sa.Column('synonyms', postgresql.JSON, nullable=False, server_default='[]'),
        sa.Column('is_obsolete', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('replaced_by', sa.String(20), nullable=True),
        sa.Column('level', sa.Integer, nullable=True, comment='Depth in GO hierarchy'),
        sa.Column('gene_count', sa.Integer, nullable=False, server_default='0',
                  comment='Number of annotated genes (including descendants)'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()'))
    )

    # Create indexes for go_terms
    op.create_index('ix_go_terms_go_id', 'go_terms', ['go_id'], unique=True)
    op.create_index('ix_go_terms_name', 'go_terms', ['name'])
    op.create_index('ix_go_terms_namespace', 'go_terms', ['namespace'])
    op.create_index('ix_go_terms_namespace_level', 'go_terms', ['namespace', 'level'])
    op.create_index('ix_go_terms_is_obsolete', 'go_terms', ['is_obsolete'])
    op.create_index('ix_go_terms_gene_count', 'go_terms', ['gene_count'])
    
    # GIN index for fuzzy text search on name
    op.execute("""
        CREATE INDEX ix_go_terms_name_search ON go_terms 
        USING gin (name gin_trgm_ops)
    """)

    # Create go_annotations table
    op.create_table(
        'go_annotations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('gene_symbol', sa.String(100), nullable=False),
        sa.Column('gene_id', sa.String(100), nullable=True),
        sa.Column('go_id', sa.String(20), nullable=False),
        sa.Column('evidence_code', sa.String(10), nullable=False,
                  comment='IEA, IDA, IMP, IGI, IPI, ISS, TAS, NAS, IC, ND, IEP, etc.'),
        sa.Column('source_db', sa.String(50), nullable=False, server_default='UniProt'),
        sa.Column('qualifier', sa.String(50), nullable=True,
                  comment='NOT, contributes_to, colocalizes_with, etc.'),
        sa.Column('organism', sa.String(100), nullable=False, server_default='Homo sapiens'),
        sa.Column('annotation_metadata', postgresql.JSON, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()'))
    )

    # Create indexes for go_annotations
    op.create_index('ix_go_annotations_gene_symbol', 'go_annotations', ['gene_symbol'])
    op.create_index('ix_go_annotations_gene_id', 'go_annotations', ['gene_id'])
    op.create_index('ix_go_annotations_go_id', 'go_annotations', ['go_id'])
    op.create_index('ix_go_annotations_evidence_code', 'go_annotations', ['evidence_code'])
    op.create_index('ix_go_annotations_organism', 'go_annotations', ['organism'])
    op.create_index('ix_go_annotations_gene_organism', 'go_annotations', ['gene_symbol', 'organism'])
    op.create_index('ix_go_annotations_go_organism', 'go_annotations', ['go_id', 'organism'])
    op.create_index('ix_go_annotations_gene_go', 'go_annotations', ['gene_symbol', 'go_id'], unique=True)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_go_annotations_gene_go', table_name='go_annotations')
    op.drop_index('ix_go_annotations_go_organism', table_name='go_annotations')
    op.drop_index('ix_go_annotations_gene_organism', table_name='go_annotations')
    op.drop_index('ix_go_annotations_organism', table_name='go_annotations')
    op.drop_index('ix_go_annotations_evidence_code', table_name='go_annotations')
    op.drop_index('ix_go_annotations_go_id', table_name='go_annotations')
    op.drop_index('ix_go_annotations_gene_id', table_name='go_annotations')
    op.drop_index('ix_go_annotations_gene_symbol', table_name='go_annotations')
    
    # Drop tables
    op.drop_table('go_annotations')
    
    # Drop go_terms indexes
    op.execute("DROP INDEX IF EXISTS ix_go_terms_name_search")
    op.drop_index('ix_go_terms_gene_count', table_name='go_terms')
    op.drop_index('ix_go_terms_is_obsolete', table_name='go_terms')
    op.drop_index('ix_go_terms_namespace_level', table_name='go_terms')
    op.drop_index('ix_go_terms_namespace', table_name='go_terms')
    op.drop_index('ix_go_terms_name', table_name='go_terms')
    op.drop_index('ix_go_terms_go_id', table_name='go_terms')
    
    # Drop table
    op.drop_table('go_terms')
