#!/usr/bin/env python3
"""
Load human GO annotations from the EBI GOA file (goa_human.gaf.gz).

Downloads the latest GOA human annotation file, parses it, and inserts
gene_symbol → go_id mappings into the go_annotations table.

Only experimental and high-quality evidence codes are imported by default.
Exclude NOT-qualified annotations.

Usage (inside the API container):
    python scripts/load_go_annotations_from_goa.py

Options:
    --evidence  Comma-separated evidence codes to include (default: EXP,IDA,IPI,IMP,IGI,IEP,HTP,HDA,HMP,HGI,HEP,IBA,IBD,IKR,IRD,ISS,ISO,ISA,ISM,IGC,IC,TAS)
    --all       Include all evidence codes including IEA (electronic)
    --limit     Limit the number of annotations to insert (for testing)
    --dry-run   Parse file but do not insert into DB
"""

import asyncio
import json
import sys
import argparse
import gzip
import io
import logging
import urllib.request
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOA_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"

# Curated + high-throughput evidence codes (excluding IEA = electronic)
DEFAULT_EVIDENCE_CODES = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",  # Experimental
    "HTP", "HDA", "HMP", "HGI", "HEP",           # High-throughput
    "IBA", "IBD", "IKR", "IRD",                   # Phylogenetically-inferred
    "ISS", "ISO", "ISA", "ISM", "IGC",            # Computational (non-IEA)
    "IC", "TAS",                                   # Author/curator statements
}

BATCH_SIZE = 5000


def parse_gaf(fileobj, evidence_codes: set, limit: int = 0) -> list[dict]:
    """Parse a GAF 2.x file and return annotation records."""
    annotations = []
    seen = set()  # (gene_symbol, go_id) pairs to deduplicate

    for raw_line in fileobj:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        cols = line.split("\t")
        if len(cols) < 15:
            continue

        qualifier = cols[3]
        # Skip NOT-qualified annotations
        if "NOT" in qualifier.upper():
            continue

        db_object_symbol = cols[2]   # Gene symbol (e.g., TP53)
        go_id = cols[4]              # GO:0000001
        evidence_code = cols[6]     # IDA, IEA, etc.
        source_db = cols[14]        # UniProtKB, etc.
        db_object_id = cols[1]      # UniProt accession

        if evidence_code not in evidence_codes:
            continue

        key = (db_object_symbol, go_id)
        if key in seen:
            continue
        seen.add(key)

        annotations.append({
            "id": str(uuid4()),
            "gene_symbol": db_object_symbol,
            "gene_id": db_object_id,
            "go_id": go_id,
            "evidence_code": evidence_code,
            "source_db": source_db.split(":")[0] if ":" in source_db else source_db,
            "qualifier": qualifier or None,
            "organism": "Homo sapiens",
            "annotation_metadata": "{}",
        })

        if limit and len(annotations) >= limit:
            break

    return annotations


async def load_annotations(
    session: AsyncSession,
    annotations: list[dict],
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(annotations)} annotations.")
        return

    logger.info("Clearing existing go_annotations …")
    await session.execute(text("DELETE FROM go_annotations"))

    logger.info(f"Inserting {len(annotations)} annotations in batches of {BATCH_SIZE} …")
    for i in range(0, len(annotations), BATCH_SIZE):
        batch = annotations[i : i + BATCH_SIZE]
        await session.execute(
            text(
                """
                INSERT INTO go_annotations
                    (id, gene_symbol, gene_id, go_id, evidence_code, source_db, qualifier, organism, annotation_metadata, created_at, updated_at)
                VALUES
                    (:id, :gene_symbol, :gene_id, :go_id, :evidence_code, :source_db, :qualifier, :organism, CAST(:annotation_metadata AS jsonb), NOW(), NOW())
                ON CONFLICT DO NOTHING
                """
            ),
            batch,
        )
        logger.info(f"  Inserted batch {i // BATCH_SIZE + 1} / {(len(annotations) + BATCH_SIZE - 1) // BATCH_SIZE}")

    await session.commit()
    logger.info("✅ Done.")


async def main(args: argparse.Namespace) -> None:
    evidence_codes = DEFAULT_EVIDENCE_CODES
    if args.all:
        evidence_codes = evidence_codes | {"IEA"}
    elif args.evidence:
        evidence_codes = set(args.evidence.split(","))

    logger.info(f"Evidence codes: {sorted(evidence_codes)}")
    logger.info(f"Downloading {GOA_URL} …")

    with urllib.request.urlopen(GOA_URL, timeout=120) as resp:
        raw = resp.read()

    logger.info(f"Downloaded {len(raw):,} bytes, parsing …")

    with gzip.open(io.BytesIO(raw), "rb") as gf:
        annotations = parse_gaf(gf, evidence_codes, limit=args.limit)

    logger.info(f"Parsed {len(annotations):,} unique (gene_symbol, go_id) pairs.")

    if args.dry_run:
        logger.info("[DRY RUN] Skipping DB insertion.")
        return

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        await load_annotations(session, annotations, dry_run=False)

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load human GO annotations from EBI GOA.")
    parser.add_argument("--evidence", default="", help="Comma-separated evidence codes to include.")
    parser.add_argument("--all", action="store_true", help="Include IEA (electronic) annotations.")
    parser.add_argument("--limit", type=int, default=0, help="Max annotations to load (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write to DB.")
    args = parser.parse_args()

    asyncio.run(main(args))
