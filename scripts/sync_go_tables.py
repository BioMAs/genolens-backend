#!/usr/bin/env python3
"""
Synchronize GO data from gene_sets to go_terms table.
This script copies GO structure from gene_sets to go_terms for use with go_service.py
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text

from app.core.config import settings
from app.models.models import GeneSet, GOTerm, GeneSetDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def sync_go_data():
    """Sync GO data from gene_sets to go_terms."""
    logger.info("Initializing DB connection...")
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get all GO gene sets
        logger.info("Fetching GO data from gene_sets...")
        query = select(GeneSet).where(
            (GeneSet.database == GeneSetDatabase.GO_BP) |
            (GeneSet.database == GeneSetDatabase.GO_MF) |
            (GeneSet.database == GeneSetDatabase.GO_CC)
        )
        result = await session.execute(query)
        gene_sets = result.scalars().all()
        
        logger.info(f"Found {len(gene_sets)} GO terms in gene_sets")
        
        # Clear existing go_terms
        logger.info("Clearing existing go_terms...")
        await session.execute(text("DELETE FROM go_terms"))
        await session.commit()
        
        # Insert into go_terms
        count = 0
        batch = []
        
        namespace_map = {
            GeneSetDatabase.GO_BP: 'biological_process',
            GeneSetDatabase.GO_MF: 'molecular_function',
            GeneSetDatabase.GO_CC: 'cellular_component'
        }
        
        for gs in gene_sets:
            metadata = gs.gene_set_metadata or {}
            
            go_term = GOTerm(
                id=uuid4(),
                go_id=gs.name,  # e.g., GO:0001234
                name=gs.description or metadata.get('definition', ''),
                namespace=namespace_map.get(gs.database, 'biological_process'),
                definition=metadata.get('description', ''),
                is_a=metadata.get('parents', []),
                part_of=[],  # Could extract from metadata if needed
                regulates=[],
                synonyms=[],
                is_obsolete=False,
                replaced_by=None,
                level=metadata.get('level', 0),
                gene_count=gs.size
            )
            
            batch.append(go_term)
            count += 1
            
            if len(batch) >= 1000:
                session.add_all(batch)
                await session.commit()
                logger.info(f"Inserted {count} GO terms...")
                batch = []
        
        # Insert remaining
        if batch:
            session.add_all(batch)
            await session.commit()
        
        logger.info(f"✅ Synced {count} GO terms from gene_sets to go_terms")
        
        # Verify
        result = await session.execute(text("SELECT COUNT(*) FROM go_terms"))
        total = result.scalar()
        logger.info(f"Total go_terms: {total}")
        
        result = await session.execute(text("SELECT namespace, COUNT(*) FROM go_terms GROUP BY namespace"))
        logger.info("Distribution by namespace:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]}")
    
    await engine.dispose()
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(sync_go_data())
