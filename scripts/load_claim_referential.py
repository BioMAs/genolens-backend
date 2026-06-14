#!/usr/bin/env python3
"""
Load Cosmetic Claim Referential Script

Imports the SciLicium "Claims_Pathway_Review_*.xlsx" workbooks bundled in
app/data/claims/ into the claim_pathway_mappings / claim_references tables and
seeds the canonical cosmetic_claims taxonomy.

Usage:
    python scripts/load_claim_referential.py
    python scripts/load_claim_referential.py --dir /path/to/Claims
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.services.claim_import_service import (  # noqa: E402
    DEFAULT_CLAIMS_DIR,
    import_claim_workbooks,
)


async def main(claims_dir: Path | None) -> None:
    paths = None
    if claims_dir is not None:
        paths = sorted(claims_dir.glob("Claims_Pathway_Review_*.xlsx"))
        if not paths:
            print(f"No workbooks found in {claims_dir}")
            sys.exit(1)

    async with AsyncSessionLocal() as db:
        result = await import_claim_workbooks(db, paths)

    print("Claim referential import complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the cosmetic claim referential")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Directory holding the workbooks (default: {DEFAULT_CLAIMS_DIR})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dir))
