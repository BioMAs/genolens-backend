"""
Initialize database with sample data for testing.
Run this after migrations to populate the database with test data.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uuid import uuid4
from app.db.session import AsyncSessionLocal
from app.models.models import Project, Sample, Dataset, DatasetType, DatasetStatus


async def init_sample_data():
    """Create sample data for testing."""
    async with AsyncSessionLocal() as db:
        try:
            # Create a test project
            test_user_id = uuid4()
            project = Project(
                name="Test Project - RNA-Seq Analysis",
                description="Sample project for testing GenoLens Next",
                owner_id=test_user_id
            )
            db.add(project)
            await db.flush()

            # Create sample metadata
            samples = [
                Sample(
                    project_id=project.id,
                    name="Control_Rep1",
                    metadata={
                        "condition": "control",
                        "replicate": 1,
                        "tissue": "liver",
                        "timepoint": "0h"
                    }
                ),
                Sample(
                    project_id=project.id,
                    name="Control_Rep2",
                    metadata={
                        "condition": "control",
                        "replicate": 2,
                        "tissue": "liver",
                        "timepoint": "0h"
                    }
                ),
                Sample(
                    project_id=project.id,
                    name="Treatment_Rep1",
                    metadata={
                        "condition": "treatment",
                        "replicate": 1,
                        "tissue": "liver",
                        "timepoint": "24h"
                    }
                ),
                Sample(
                    project_id=project.id,
                    name="Treatment_Rep2",
                    metadata={
                        "condition": "treatment",
                        "replicate": 2,
                        "tissue": "liver",
                        "timepoint": "24h"
                    }
                ),
            ]

            for sample in samples:
                db.add(sample)

            await db.commit()

            print(f"✅ Sample data created successfully!")
            print(f"📁 Project ID: {project.id}")
            print(f"👤 Test User ID: {test_user_id}")
            print(f"📊 Created {len(samples)} samples")
            print(f"\n⚠️  Note: Use the Test User ID for authentication in your tests")

        except Exception as e:
            await db.rollback()
            print(f"❌ Error creating sample data: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(init_sample_data())
