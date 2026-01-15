#!/usr/bin/env python3
"""
Load Gene Ontology Structure Script.
Downloads and parses the latest go-basic.obo file.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.services.go_loader import GoLoaderService

logging.basicConfig(level=logging.INFO)

async def main():
    print("Initialize DB connection...")
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        loader = GoLoaderService(session)
        
        url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
        print(f"Loading GO ontology from {url}...")
        
        await loader.load_obo(url=url)
        
    await engine.dispose()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
