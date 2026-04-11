"""One-shot script: add MATRIX dataset to all existing demo projects.

Run inside the container:
    docker exec backend-api-1 python3 /app/scripts/patch_demo_matrix.py
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select


async def main() -> None:
    # Import here so the script works from the app root
    from app.db.session import AsyncSessionLocal
    from app.models.models import Project
    from app.services.demo_seed_service import (
        DEMO_PROJECT_NAME,
        add_matrix_to_existing_demo_project,
        add_sample_metadata_to_existing_demo_project,
    )

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Project).where(Project.name == DEMO_PROJECT_NAME)
        )
        projects = result.scalars().all()

        if not projects:
            print(f"No demo projects found with name '{DEMO_PROJECT_NAME}'")
            sys.exit(0)

        print(f"Found {len(projects)} demo project(s)")
        for p in projects:
            dataset = await add_matrix_to_existing_demo_project(db, p.id)
            print(f"  -> Project {p.id}: MATRIX dataset {dataset.id}")
            meta = await add_sample_metadata_to_existing_demo_project(db, p.id)
            print(f"  -> Project {p.id}: METADATA_SAMPLE dataset {meta.id}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
