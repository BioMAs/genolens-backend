"""
Service for loading Gene Ontology structure from OBO files.
"""
from pathlib import Path
from typing import Dict, Any, List
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
import requests
import tempfile
import os

from goatools.obo_parser import GODag

from app.models.models import GeneSet, GeneSetDatabase

logger = logging.getLogger(__name__)

class GoLoaderService:
    """Service to load GO ontology structure."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def load_obo(self, obo_path: str = None, url: str = None):
        """
        Load GO IDs, names, definitions and hierarchy from OBO file.
        Updates existing GeneSets or creates new empty ones (without genes initially).
        """
        temp_file = None
        if url and not obo_path:
            logger.info(f"Downloading OBO from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            
            fd, temp_path = tempfile.mkstemp(suffix=".obo")
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)
            obo_path = temp_path
            temp_file = temp_path

        try:
            logger.info(f"Parsing OBO file: {obo_path}")
            godag = GODag(obo_path, optional_attrs={'def'})
            
            logger.info(f"Loaded {len(godag)} GO terms. Updating database...")
            
            count = 0
            for go_id, term in godag.items():
                if term.is_obsolete:
                    continue
                
                # Determine Database type
                namespace_map = {
                    'biological_process': GeneSetDatabase.GO_BP,
                    'molecular_function': GeneSetDatabase.GO_MF,
                    'cellular_component': GeneSetDatabase.GO_CC
                }
                
                db_type = namespace_map.get(term.namespace)
                if not db_type:
                    continue

                # Prepare metadata
                metadata = {
                    "go_id": go_id,
                    "definition": term.name, # Term name e.g. "mitochondrion"
                    "description": term.defn, # Long textual definition
                    "parents": [p.id for p in term.parents],
                    "children": [c.id for c in term.children],
                    "level": term.level,
                    "namespace": term.namespace
                }

                # We use the GO ID as the name for reliability, or we can use the term name.
                # Usually GSEA uses names like "GOBP_MITOCHONDRIAL_TRANSLATION".
                # Standard GO use IDs like "GO:0000001".
                # To support both, we might need a mapping or check if we update existing by name or ID.
                # Strategy: We treat GO:ID as the canonical record for the Browser.
                # If we have existing GSEA sets, they often have GO IDs in their metadata or description.
                # For this "GO Browser" feature, let's store them as "GO:XXXXXXX" name.
                
                stmt = insert(GeneSet).values(
                    name=go_id, # GO:0001234
                    database=db_type,
                    description=term.name, # "mitochondrion" (short name as description for list view)
                    genes=[], # We don't have gene associations in OBO, only structure
                    size=0,
                    gene_set_metadata=metadata,
                    organism="All" # Ontology is species agnostic
                ).on_conflict_do_update(
                    index_elements=['name', 'database'],
                    set_={
                        "description": term.name,
                        "gene_set_metadata": metadata 
                        # We preserve existing genes/size if it existed
                    }
                )
                
                await self.db.execute(stmt)
                count += 1
                
                if count % 1000 == 0:
                    await self.db.commit()
                    logger.info(f"Processed {count} terms...")

            await self.db.commit()
            logger.info(f"Finished loading {count} GO terms.")

        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

