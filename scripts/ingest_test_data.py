import asyncio
import os
import sys
from pathlib import Path
import jwt
import httpx
from datetime import datetime, timedelta
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings

# Configuration
API_URL = "http://localhost:8000/api/v1"
DATA_DIR = Path("data_test/Custom_analysis")

# File mapping to DatasetType
FILE_MAPPING = {
    "counts.xlsx": "MATRIX",
    "rlog_data.xlsx": "MATRIX",
    "export_table.xlsx": "DEG",
    "funct.analysis_STRATEGY1_data_p0.05_r3_onlyenriched.xlsx": "ENRICHMENT",
    "samples_info_20251120.xlsx": "METADATA",
    "contrasts_info_20251120.xlsx": "METADATA"
}

def generate_test_token(user_id: str) -> str:
    """Generate a valid JWT token for testing."""
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "exp": datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, settings.SUPABASE_JWT_SECRET, algorithm="HS256")

async def ingest_data():
    print("🚀 Starting data ingestion...")
    
    # 1. Generate Token
    # Use the REAL user ID for test@genolens.com so we can see data in the login test
    user_id = "a4ca0137-6779-46b1-8100-59f11dc009cc" 
    token = generate_test_token(user_id)
    headers = {"Authorization": f"Bearer {token}"}
    print(f"✓ Generated test token for user {user_id}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 2. Create Project
        print("\n📦 Creating Project...")
        project_data = {
            "name": "Test Analysis 2025",
            "description": "Imported from Custom_analysis folder"
        }
        response = await client.post(f"{API_URL}/projects/", json=project_data, headers=headers)
        
        if response.status_code != 201:
            print(f"❌ Failed to create project: {response.text}")
            return
        
        project = response.json()
        project_id = project["id"]
        print(f"✓ Project created: {project['name']} (ID: {project_id})")

        # 3. Upload Files
        print("\nCc Uploading Files...")
        for filename, dataset_type in FILE_MAPPING.items():
            file_path = DATA_DIR / filename
            if not file_path.exists():
                print(f"⚠️ File not found: {filename}")
                continue

            print(f"  • Uploading {filename} as {dataset_type}...")
            
            files = {
                "file": (filename, open(file_path, "rb"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            }
            data = {
                "project_id": project_id,
                "name": filename.split(".")[0],
                "dataset_type": dataset_type,
                "description": f"Imported {dataset_type} file"
            }

            try:
                response = await client.post(
                    f"{API_URL}/datasets/upload",
                    data=data,
                    files=files,
                    headers=headers
                )
                
                if response.status_code == 201:
                    print(f"    ✓ Success")
                else:
                    print(f"    ❌ Failed: {response.text}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")

    print("\n✨ Ingestion complete!")
    print(f"👉 You can now query the API for project {project_id}")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        exit(1)
        
    asyncio.run(ingest_data())
