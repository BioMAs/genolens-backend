#!/usr/bin/env python3
"""
CLI script — assign demo data to a user.

Usage:
    cd backend/
    python scripts/seed_demo_data.py --user-id <UUID>

The script connects directly to the database (using the same settings as the app)
and creates a demo project with 2 DEG comparisons for the given user.
"""
import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

# Ensure backend/ is on the Python path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal
from app.services.demo_seed_service import DemoAlreadyExistsError, create_demo_data


async def main(user_id: UUID) -> None:
    async with AsyncSessionLocal() as db:
        print(f"Creating demo data for user {user_id} …")
        try:
            project = await create_demo_data(db, user_id)
        except DemoAlreadyExistsError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    print()
    print("✓ Demo data created successfully")
    print(f"  Project ID  : {project.id}")
    print(f"  Project name: {project.name}")
    print()
    print("Datasets:")
    for ds in project.datasets:
        print(
            f"  [{ds.id}]  {ds.name}  —  "
            f"↑{ds.deg_up_count} ↓{ds.deg_down_count}  "
            f"({ds.deg_significant_count}/{ds.total_genes} significant)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo data for a test user")
    parser.add_argument(
        "--user-id",
        required=True,
        help="Supabase Auth user UUID to assign the demo project to",
    )
    args = parser.parse_args()

    try:
        uid = UUID(args.user_id)
    except ValueError:
        print(f"ERROR: '{args.user_id}' is not a valid UUID")
        sys.exit(1)

    asyncio.run(main(uid))
