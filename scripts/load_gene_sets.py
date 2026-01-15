#!/usr/bin/env python3
"""
Load Gene Sets Script

This script loads gene sets from GMT files into the database.
Supports MSigDB, GO, KEGG, and other gene set databases.

Usage:
    python scripts/load_gene_sets.py --file path/to/geneset.gmt --database GO_BP --organism "Homo sapiens"
    python scripts/load_gene_sets.py --file hallmark.gmt --database HALLMARK --clear
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.services.gene_set_loader import GeneSetLoader, validate_gmt_file
from app.models.models import GeneSetDatabase


async def load_gene_sets(
    file_path: str,
    database: str,
    organism: str = "Homo sapiens",
    version: str = None,
    clear_existing: bool = False
):
    """Load gene sets from GMT file into database."""

    # Validate file
    print(f"Validating GMT file: {file_path}")
    validation = validate_gmt_file(file_path)

    if not validation['valid']:
        print(f"❌ Validation failed: {validation['error']}")
        return 1

    print(f"✅ Validation successful:")
    print(f"   - Gene sets: {validation['gene_set_count']}")
    print(f"   - Size range: {validation['min_size']}-{validation['max_size']}")
    print(f"   - Avg size: {validation['avg_size']:.1f}")
    print(f"   - Unique genes: {validation['unique_genes']}")

    # Convert database string to enum
    try:
        database_enum = GeneSetDatabase(database)
    except ValueError:
        print(f"❌ Invalid database: {database}")
        print(f"Valid options: {[e.value for e in GeneSetDatabase]}")
        return 1

    # Create database session
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        loader = GeneSetLoader(session)

        # Load gene sets
        print(f"\nLoading gene sets into database...")
        print(f"   Database: {database}")
        print(f"   Organism: {organism}")
        if version:
            print(f"   Version: {version}")
        if clear_existing:
            print(f"   ⚠️  Clearing existing gene sets for {database}/{organism}")

        inserted, skipped = await loader.load_from_gmt(
            file_path=file_path,
            database=database_enum,
            organism=organism,
            version=version,
            clear_existing=clear_existing
        )

        print(f"\n✅ Loading complete:")
        print(f"   - Inserted: {inserted}")
        print(f"   - Updated: {skipped}")

        # Show database stats
        print(f"\n📊 Database statistics:")
        stats = await loader.get_database_stats()
        for db_name, count in sorted(stats.items()):
            if count > 0:
                print(f"   - {db_name}: {count} gene sets")

    await engine.dispose()
    return 0


async def search_gene_sets(query: str, database: str = None, limit: int = 20):
    """Search gene sets in the database."""
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        loader = GeneSetLoader(session)

        database_enum = None
        if database:
            try:
                database_enum = GeneSetDatabase(database)
            except ValueError:
                print(f"❌ Invalid database: {database}")
                return 1

        results = await loader.search_gene_sets(
            query=query,
            database=database_enum,
            limit=limit
        )

        print(f"\n🔍 Search results for '{query}':")
        print(f"Found {len(results)} gene sets:\n")

        for gs in results:
            print(f"  {gs.name}")
            print(f"    Database: {gs.database.value}")
            print(f"    Size: {gs.size} genes")
            if gs.description and gs.description != gs.name:
                desc = gs.description[:100] + "..." if len(gs.description) > 100 else gs.description
                print(f"    Description: {desc}")
            print()

    await engine.dispose()
    return 0


async def show_stats():
    """Show database statistics."""
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        loader = GeneSetLoader(session)

        print("\n📊 Gene Set Database Statistics:\n")
        stats = await loader.get_database_stats()

        total = sum(stats.values())
        print(f"Total gene sets: {total}\n")

        for db_name, count in sorted(stats.items()):
            bar = "█" * min(50, int(count / max(1, total / 50)))
            print(f"  {db_name:20s} {count:6d} {bar}")

    await engine.dispose()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Load gene sets into GenoLens database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load MSigDB Hallmark gene sets
  python scripts/load_gene_sets.py --file h.all.v2024.1.Hs.symbols.gmt --database HALLMARK --version 2024.1

  # Load GO Biological Process (clear existing)
  python scripts/load_gene_sets.py --file c5.go.bp.v2024.1.Hs.symbols.gmt --database GO_BP --clear

  # Search for gene sets
  python scripts/load_gene_sets.py --search "TNFA" --database HALLMARK

  # Show statistics
  python scripts/load_gene_sets.py --stats
        """
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to GMT file"
    )
    parser.add_argument(
        "--database",
        type=str,
        choices=[e.value for e in GeneSetDatabase],
        help="Gene set database type"
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Homo sapiens",
        help="Organism/species (default: Homo sapiens)"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Version of gene set database (e.g., 2024.1)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing gene sets for this database/organism before loading"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search for gene sets by name or description"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum search results (default: 20)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )

    args = parser.parse_args()

    # Determine which action to perform
    if args.stats:
        return asyncio.run(show_stats())

    elif args.search:
        return asyncio.run(search_gene_sets(
            query=args.search,
            database=args.database,
            limit=args.limit
        ))

    elif args.file and args.database:
        return asyncio.run(load_gene_sets(
            file_path=args.file,
            database=args.database,
            organism=args.organism,
            version=args.version,
            clear_existing=args.clear
        ))

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
