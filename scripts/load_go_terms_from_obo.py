#!/usr/bin/env python3
"""
Load GO terms into the go_terms table from the go-basic.obo file.

Downloads the latest OBO file from OBO Foundry and inserts GOTerm records.
This script targets the go_terms table (used by GOService for enrichment analysis),
NOT the gene_sets table (which is loaded by load_go_ontology.py / go_loader.py).

Usage (inside the API container):
    python scripts/load_go_terms_from_obo.py
    python scripts/load_go_terms_from_obo.py --dry-run
"""

import asyncio
import sys
import argparse
import gzip
import io
import json
import logging
import re
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

OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
BATCH_SIZE = 2000

NAMESPACE_MAP = {
    "biological_process": "biological_process",
    "molecular_function": "molecular_function",
    "cellular_component": "cellular_component",
}


def parse_obo(text_content: str) -> list[dict]:
    """Parse OBO 1.2 format and return list of term dicts for go_terms table."""
    terms = []
    current = None

    for line in text_content.splitlines():
        line = line.strip()

        if line == "[Term]":
            if current and current.get("go_id") and not current.get("is_obsolete", False):
                terms.append(current)
            current = {
                "is_a": [],
                "part_of": [],
                "regulates": [],
                "synonyms": [],
                "is_obsolete": False,
                "replaced_by": None,
                "level": None,
                "gene_count": 0,
            }
            continue

        if line == "[Typedef]" or (line.startswith("[") and line != "[Term]"):
            # Save pending term and reset
            if current and current.get("go_id") and not current.get("is_obsolete", False):
                terms.append(current)
            current = None
            continue

        if current is None:
            continue

        if line.startswith("id: GO:"):
            current["go_id"] = line[4:].strip()
        elif line.startswith("name: "):
            current["name"] = line[6:].strip()
        elif line.startswith("namespace: "):
            current["namespace"] = line[11:].strip()
        elif line.startswith("def: "):
            # def: "text" [refs]  → extract just the text
            m = re.match(r'def: "(.+?)"', line)
            current["definition"] = m.group(1) if m else ""
        elif line.startswith("is_a: "):
            # is_a: GO:0000002 ! name
            m = re.match(r"is_a: (GO:\d+)", line)
            if m:
                current["is_a"].append(m.group(1))
        elif line.startswith("relationship: part_of "):
            m = re.match(r"relationship: part_of (GO:\d+)", line)
            if m:
                current["part_of"].append(m.group(1))
        elif line.startswith("relationship: regulates ") or \
             line.startswith("relationship: negatively_regulates ") or \
             line.startswith("relationship: positively_regulates "):
            m = re.match(r"relationship: \S+ (GO:\d+)", line)
            if m:
                current["regulates"].append(m.group(1))
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.+?)"', line)
            if m:
                current["synonyms"].append(m.group(1))
        elif line == "is_obsolete: true":
            current["is_obsolete"] = True
        elif line.startswith("replaced_by: "):
            m = re.match(r"replaced_by: (GO:\d+)", line)
            if m:
                current["replaced_by"] = m.group(1)

    # Don't forget the last term
    if current and current.get("go_id") and not current.get("is_obsolete", False):
        terms.append(current)

    return terms


def compute_levels(terms: list[dict]) -> dict[str, int]:
    """BFS from roots to compute level (depth) for each GO term."""
    by_id = {t["go_id"]: t for t in terms}
    children: dict[str, set] = {t["go_id"]: set() for t in terms}

    for term in terms:
        for parent_id in term["is_a"] + term["part_of"]:
            if parent_id in children:
                children[parent_id].add(term["go_id"])

    # Roots = terms with no parents inside this set
    roots = {t["go_id"] for t in terms if not any(p in by_id for p in t["is_a"] + t["part_of"])}

    levels: dict[str, int] = {}
    queue = list(roots)
    for go_id in queue:
        levels[go_id] = 0

    visited = set(roots)
    while queue:
        current_id = queue.pop(0)
        for child_id in children.get(current_id, []):
            if child_id not in visited:
                levels[child_id] = levels[current_id] + 1
                visited.add(child_id)
                queue.append(child_id)

    return levels


async def load_terms(session: AsyncSession, terms: list[dict], dry_run: bool = False) -> None:
    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(terms)} GO terms.")
        return

    logger.info("Clearing existing go_terms …")
    await session.execute(text("DELETE FROM go_terms"))
    await session.commit()

    logger.info(f"Inserting {len(terms)} GO terms in batches of {BATCH_SIZE} …")
    for i in range(0, len(terms), BATCH_SIZE):
        batch = terms[i: i + BATCH_SIZE]
        await session.execute(
            text(
                """
                INSERT INTO go_terms
                    (id, go_id, name, namespace, definition,
                     is_a, part_of, regulates, synonyms,
                     is_obsolete, replaced_by, level, gene_count,
                     created_at, updated_at)
                VALUES
                    (:id, :go_id, :name, :namespace, :definition,
                     CAST(:is_a AS jsonb), CAST(:part_of AS jsonb),
                     CAST(:regulates AS jsonb), CAST(:synonyms AS jsonb),
                     :is_obsolete, :replaced_by, :level, :gene_count,
                     NOW(), NOW())
                ON CONFLICT (go_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    namespace = EXCLUDED.namespace,
                    definition = EXCLUDED.definition,
                    is_a = EXCLUDED.is_a,
                    part_of = EXCLUDED.part_of,
                    regulates = EXCLUDED.regulates,
                    synonyms = EXCLUDED.synonyms,
                    level = EXCLUDED.level,
                    updated_at = NOW()
                """
            ),
            [
                {
                    **t,
                    "id": str(uuid4()),
                    "is_a": json.dumps(t["is_a"]),
                    "part_of": json.dumps(t["part_of"]),
                    "regulates": json.dumps(t["regulates"]),
                    "synonyms": json.dumps(t["synonyms"]),
                    "definition": t.get("definition", ""),
                }
                for t in batch
            ],
        )
        await session.commit()
        logger.info(f"  Batch {i // BATCH_SIZE + 1} / {(len(terms) + BATCH_SIZE - 1) // BATCH_SIZE}")

    logger.info("✅ Done.")


async def main(args: argparse.Namespace) -> None:
    logger.info(f"Downloading {OBO_URL} …")
    with urllib.request.urlopen(OBO_URL, timeout=120) as resp:
        content = resp.read().decode("utf-8")

    logger.info(f"Downloaded {len(content):,} bytes, parsing …")
    terms = parse_obo(content)
    logger.info(f"Parsed {len(terms)} non-obsolete GO terms.")

    logger.info("Computing term levels (BFS) …")
    levels = compute_levels(terms)
    for term in terms:
        term["level"] = levels.get(term["go_id"])

    # Namespace breakdown
    from collections import Counter
    ns_counts = Counter(t["namespace"] for t in terms)
    for ns, count in sorted(ns_counts.items()):
        logger.info(f"  {ns}: {count:,} terms")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        await load_terms(session, terms, dry_run=args.dry_run)

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load GO terms into go_terms table from OBO")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not insert")
    args = parser.parse_args()
    asyncio.run(main(args))
