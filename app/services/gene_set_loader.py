"""
Gene Set Loader Service

Handles loading gene sets from various sources:
- GMT files (MSigDB format)
- GO OBO files
- KEGG API
- Custom user uploads

GMT Format:
Each line: <gene_set_name>\t<description>\t<gene1>\t<gene2>\t...
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from uuid import UUID
import logging

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import GeneSet, GeneSetDatabase

logger = logging.getLogger(__name__)


class GMTParser:
    """Parser for GMT (Gene Matrix Transposed) files."""

    @staticmethod
    def parse_file(file_path: str) -> List[Dict]:
        """
        Parse a GMT file and return a list of gene sets.

        Args:
            file_path: Path to the GMT file

        Returns:
            List of dictionaries containing gene set information

        Example GMT line:
        HALLMARK_TNFA_SIGNALING_VIA_NFKB	http://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_TNFA_SIGNALING_VIA_NFKB	ABCA1	ABI1	ACKR3	...
        """
        gene_sets = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    parts = line.split('\t')
                    if len(parts) < 3:
                        logger.warning(f"Line {line_num}: Invalid format (less than 3 fields)")
                        continue

                    name = parts[0]
                    description = parts[1] if parts[1] and not parts[1].startswith('http') else name
                    genes = [g.strip() for g in parts[2:] if g.strip()]

                    # Skip empty gene sets
                    if not genes:
                        logger.warning(f"Line {line_num}: Gene set '{name}' has no genes")
                        continue

                    gene_sets.append({
                        'name': name,
                        'description': description,
                        'genes': genes,
                        'size': len(genes)
                    })

                except Exception as e:
                    logger.error(f"Line {line_num}: Error parsing - {str(e)}")
                    continue

        logger.info(f"Parsed {len(gene_sets)} gene sets from {file_path}")
        return gene_sets


class GeneSetLoader:
    """Service for loading gene sets into the database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def load_from_gmt(
        self,
        file_path: str,
        database: GeneSetDatabase,
        organism: str = "Homo sapiens",
        version: Optional[str] = None,
        clear_existing: bool = False
    ) -> Tuple[int, int]:
        """
        Load gene sets from a GMT file.

        Args:
            file_path: Path to the GMT file
            database: Which database these gene sets belong to
            organism: Organism/species
            version: Version of the gene set database
            clear_existing: If True, delete existing gene sets for this database/organism

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        # Parse the GMT file
        gene_sets_data = GMTParser.parse_file(file_path)

        # Clear existing gene sets if requested
        if clear_existing:
            await self._clear_existing(database, organism)

        # Load gene sets into database
        inserted = 0
        skipped = 0

        for gs_data in gene_sets_data:
            try:
                # Check if gene set already exists
                stmt = select(GeneSet).where(
                    GeneSet.name == gs_data['name'],
                    GeneSet.database == database,
                    GeneSet.organism == organism
                )
                result = await self.db.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing gene set
                    existing.description = gs_data['description']
                    existing.genes = gs_data['genes']
                    existing.size = gs_data['size']
                    existing.version = version
                    skipped += 1
                else:
                    # Insert new gene set
                    gene_set = GeneSet(
                        name=gs_data['name'],
                        database=database,
                        description=gs_data['description'],
                        genes=gs_data['genes'],
                        size=gs_data['size'],
                        organism=organism,
                        version=version,
                        gene_set_metadata={}
                    )
                    self.db.add(gene_set)
                    inserted += 1

                # Commit in batches of 100 to avoid memory issues
                if (inserted + skipped) % 100 == 0:
                    await self.db.commit()
                    logger.info(f"Processed {inserted + skipped} gene sets...")

            except Exception as e:
                logger.error(f"Error loading gene set '{gs_data['name']}': {str(e)}")
                continue

        # Final commit
        await self.db.commit()

        logger.info(f"Loaded {inserted} gene sets, updated {skipped} existing")
        return inserted, skipped

    async def _clear_existing(self, database: GeneSetDatabase, organism: str):
        """Clear existing gene sets for a database/organism combination."""
        stmt = delete(GeneSet).where(
            GeneSet.database == database,
            GeneSet.organism == organism
        )
        await self.db.execute(stmt)
        await self.db.commit()
        logger.info(f"Cleared existing gene sets for {database}/{organism}")

    async def get_gene_sets(
        self,
        database: GeneSetDatabase,
        organism: str = "Homo sapiens",
        min_size: int = 15,
        max_size: int = 500
    ) -> List[GeneSet]:
        """
        Retrieve gene sets from the database.

        Args:
            database: Which database to query
            organism: Organism/species
            min_size: Minimum gene set size
            max_size: Maximum gene set size

        Returns:
            List of GeneSet objects
        """
        stmt = select(GeneSet).where(
            GeneSet.database == database,
            GeneSet.organism == organism,
            GeneSet.size >= min_size,
            GeneSet.size <= max_size
        ).order_by(GeneSet.name)

        result = await self.db.execute(stmt)
        gene_sets = result.scalars().all()

        logger.info(f"Retrieved {len(gene_sets)} gene sets for {database}/{organism}")
        return list(gene_sets)

    async def get_gene_set_by_name(
        self,
        name: str,
        database: GeneSetDatabase,
        organism: str = "Homo sapiens"
    ) -> Optional[GeneSet]:
        """Get a specific gene set by name."""
        stmt = select(GeneSet).where(
            GeneSet.name == name,
            GeneSet.database == database,
            GeneSet.organism == organism
        )

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about loaded gene sets."""
        stats = {}

        for database in GeneSetDatabase:
            stmt = select(GeneSet).where(GeneSet.database == database)
            result = await self.db.execute(stmt)
            count = len(result.scalars().all())
            stats[database.value] = count

        return stats

    async def search_gene_sets(
        self,
        query: str,
        database: Optional[GeneSetDatabase] = None,
        organism: str = "Homo sapiens",
        limit: int = 50
    ) -> List[GeneSet]:
        """
        Search gene sets by name or description.

        Args:
            query: Search term
            database: Optional database filter
            organism: Organism/species
            limit: Maximum number of results

        Returns:
            List of matching GeneSet objects
        """
        stmt = select(GeneSet).where(
            GeneSet.organism == organism
        )

        if database:
            stmt = stmt.where(GeneSet.database == database)

        # Search in name or description (case-insensitive)
        search_pattern = f"%{query}%"
        stmt = stmt.where(
            (GeneSet.name.ilike(search_pattern)) |
            (GeneSet.description.ilike(search_pattern))
        )

        stmt = stmt.limit(limit).order_by(GeneSet.name)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())


# Utility functions for working with gene set files

def validate_gmt_file(file_path: str) -> Dict[str, any]:
    """
    Validate a GMT file format and return summary statistics.

    Args:
        file_path: Path to the GMT file

    Returns:
        Dictionary with validation results and statistics
    """
    if not Path(file_path).exists():
        return {
            'valid': False,
            'error': f"File not found: {file_path}"
        }

    try:
        gene_sets = GMTParser.parse_file(file_path)

        if not gene_sets:
            return {
                'valid': False,
                'error': "No valid gene sets found in file"
            }

        sizes = [gs['size'] for gs in gene_sets]

        return {
            'valid': True,
            'gene_set_count': len(gene_sets),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': sum(sizes) / len(sizes),
            'total_genes': sum(sizes),
            'unique_genes': len(set([g for gs in gene_sets for g in gs['genes']]))
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def convert_gene_symbols(
    genes: List[str],
    from_format: str = "symbol",
    to_format: str = "ensembl"
) -> Dict[str, str]:
    """
    Convert gene identifiers between different formats.

    Note: This is a placeholder. In production, you would use a service like
    MyGene.info API or a local mapping database.

    Args:
        genes: List of gene identifiers
        from_format: Source format (symbol, ensembl, entrez)
        to_format: Target format

    Returns:
        Dictionary mapping source to target IDs
    """
    # Placeholder - in production, implement actual conversion
    logger.warning("Gene ID conversion not yet implemented - returning identity mapping")
    return {gene: gene for gene in genes}
