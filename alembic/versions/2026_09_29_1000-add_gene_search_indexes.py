"""index deg_genes for case-insensitive gene prefix search

- ix_deg_genes_gene_name_upper_pattern : upper(gene_name) text_pattern_ops
- ix_deg_genes_gene_id_upper_pattern   : upper(gene_id)   text_pattern_ops

`/genes/search` filtre `upper(gene_name) LIKE 'X%' OR upper(gene_id) LIKE 'X%'`. Sans
ces index, chaque frappe dans la palette de commandes balaie toute la table : sur 6 M
de lignes (300 comparaisons de 20 000 gènes), 480 ms par requête, et linéaire en volume.
Avec eux, un BitmapOr des deux index : 0,4 ms pour « TP53 », 2 ms pour « TP ».
`text_pattern_ops` est nécessaire parce que l'index btree par défaut ne sert LIKE que
sous la collation C.

Construits CONCURRENTLY, hors transaction : `deg_genes` est la plus grosse table, et un
CREATE INDEX ordinaire y bloquerait les écritures d'ingestion le temps du build.

Idempotent (IF NOT EXISTS), comme les migrations voisines.

Revision ID: gene_search_indexes_001
Revises: expiry_warning_sent_001
Create Date: 2026-09-29 10:00:00
"""

from typing import Sequence, Union

from alembic import op

revision: str = "gene_search_indexes_001"
down_revision: Union[str, None] = "expiry_warning_sent_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEXES = {
    "ix_deg_genes_gene_name_upper_pattern": "upper(gene_name) text_pattern_ops",
    "ix_deg_genes_gene_id_upper_pattern": "upper(gene_id) text_pattern_ops",
}


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, expression in INDEXES.items():
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON deg_genes ({expression})"
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
