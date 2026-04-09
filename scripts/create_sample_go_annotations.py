#!/usr/bin/env python3
"""
Create sample GO annotations for testing.
Adds annotations for common genes to enable GO enrichment testing.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.core.config import settings
from app.models.models import GOAnnotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample annotations for common genes
SAMPLE_ANNOTATIONS = [
    # TP53 - tumor suppressor, cell cycle, apoptosis
    ("TP53", "P04637", "GO:0006915", "IDA", "apoptotic process"),  # BP
    ("TP53", "P04637", "GO:0008283", "IDA", "cell population proliferation"),  # BP
    ("TP53", "P04637", "GO:0042981", "IDA", "regulation of apoptotic process"),  # BP
    ("TP53", "P04637", "GO:0003677", "IDA", "DNA binding"),  # MF
    ("TP53", "P04637", "GO:0005634", "IDA", "nucleus"),  # CC
    
    # BCL2 - anti-apoptotic
    ("BCL2", "P10415", "GO:0006915", "IDA", "apoptotic process"),
    ("BCL2", "P10415", "GO:0043066", "IDA", "negative regulation of apoptotic process"),
    ("BCL2", "P10415", "GO:0005741", "IDA", "mitochondrial outer membrane"),
    
    # CASP3 - caspase, apoptosis executor
    ("CASP3", "P42574", "GO:0006915", "IDA", "apoptotic process"),
    ("CASP3", "P42574", "GO:0097194", "IDA", "execution phase of apoptosis"),
    ("CASP3", "P42574", "GO:0004197", "IDA", "cysteine-type endopeptidase activity"),
    ("CASP3", "P42574", "GO:0005737", "IDA", "cytoplasm"),
    
    # MYC - oncogene, cell proliferation
    ("MYC", "P01106", "GO:0008283", "IDA", "cell population proliferation"),
    ("MYC", "P01106", "GO:0006355", "IDA", "regulation of transcription"),
    ("MYC", "P01106", "GO:0003700", "IDA", "DNA-binding transcription factor activity"),
    ("MYC", "P01106", "GO:0005634", "IDA", "nucleus"),
    
    # BRCA1 - DNA repair
    ("BRCA1", "P38398", "GO:0006281", "IDA", "DNA repair"),
    ("BRCA1", "P38398", "GO:0006974", "IDA", "cellular response to DNA damage stimulus"),
    ("BRCA1", "P38398", "GO:0003677", "IDA", "DNA binding"),
    ("BRCA1", "P38398", "GO:0005634", "IDA", "nucleus"),
    
    # Add more common genes...
    ("EGFR", "P00533", "GO:0007165", "IDA", "signal transduction"),
    ("EGFR", "P00533", "GO:0004714", "IDA", "transmembrane receptor protein tyrosine kinase activity"),
    ("EGFR", "P00533", "GO:0005886", "IDA", "plasma membrane"),
    
    ("KRAS", "P01116", "GO:0007165", "IDA", "signal transduction"),
    ("KRAS", "P01116", "GO:0003924", "IDA", "GTPase activity"),
    ("KRAS", "P01116", "GO:0005886", "IDA", "plasma membrane"),
    
    ("AKT1", "P31749", "GO:0006468", "IDA", "protein phosphorylation"),
    ("AKT1", "P31749", "GO:0004674", "IDA", "protein serine/threonine kinase activity"),
    ("AKT1", "P31749", "GO:0005737", "IDA", "cytoplasm"),
    
    ("PTEN", "P60484", "GO:0008285", "IDA", "negative regulation of cell population proliferation"),
    ("PTEN", "P60484", "GO:0004725", "IDA", "protein tyrosine phosphatase activity"),
    ("PTEN", "P60484", "GO:0005737", "IDA", "cytoplasm"),
    
    ("TNF", "P01375", "GO:0006954", "IDA", "inflammatory response"),
    ("TNF", "P01375", "GO:0005125", "IDA", "cytokine activity"),
    ("TNF", "P01375", "GO:0005615", "IDA", "extracellular space"),
]


async def create_sample_annotations():
    """Create sample GO annotations for testing."""
    logger.info("Initializing DB connection...")
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Clear existing annotations
        logger.info("Clearing existing go_annotations...")
        await session.execute(text("DELETE FROM go_annotations"))
        await session.commit()
        
        # Verify GO terms exist
        logger.info("Verifying GO terms...")
        result = await session.execute(text("SELECT COUNT(*) FROM go_terms"))
        term_count = result.scalar()
        if term_count == 0:
            logger.error("No GO terms found! Run load_go_ontology.py first.")
            return
        
        logger.info(f"Found {term_count} GO terms")
        
        # Create annotations
        logger.info(f"Creating {len(SAMPLE_ANNOTATIONS)} sample annotations...")
        annotations = []
        
        for gene_symbol, gene_id, go_id, evidence, description in SAMPLE_ANNOTATIONS:
            annotation = GOAnnotation(
                id=uuid4(),
                gene_symbol=gene_symbol,
                gene_id=gene_id,
                go_id=go_id,
                evidence_code=evidence,
                source_db="UniProt",
                qualifier=None,
                organism="Homo sapiens",
                annotation_metadata={"description": description}
            )
            annotations.append(annotation)
        
        session.add_all(annotations)
        await session.commit()
        
        logger.info(f"✅ Created {len(annotations)} sample GO annotations")
        
        # Verify
        result = await session.execute(text("SELECT COUNT(*) FROM go_annotations"))
        total = result.scalar()
        logger.info(f"Total go_annotations: {total}")
        
        # Show distribution
        result = await session.execute(text("""
            SELECT gene_symbol, COUNT(*) as count 
            FROM go_annotations 
            GROUP BY gene_symbol 
            ORDER BY count DESC
        """))
        logger.info("Annotations per gene:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]} terms")
    
    await engine.dispose()
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(create_sample_annotations())
