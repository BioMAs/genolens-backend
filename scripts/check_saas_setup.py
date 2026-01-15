#!/usr/bin/env python3
"""Script pour vérifier la configuration SaaS"""
import asyncio
from sqlalchemy import text
from app.db.session import engine


async def check_tables():
    """Vérifier les tables créées"""
    async with engine.connect() as conn:
        # Lister les tables
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        )
        tables = [row[0] for row in result]
        print("📊 Tables disponibles:")
        for table in tables:
            print(f"  - {table}")
        
        # Vérifier l'utilisateur admin
        print("\n👤 Utilisateur admin:")
        result = await conn.execute(
            text("SELECT id, email, role, subscription_tier, max_projects FROM users WHERE role = 'ADMIN'")
        )
        for row in result:
            print(f"  ID: {row[0]}")
            print(f"  Email: {row[1]}")
            print(f"  Role: {row[2]}")
            print(f"  Tier: {row[3]}")
            print(f"  Max Projects: {row[4]}")
        
        # Compter les projets
        print("\n📁 Projets:")
        result = await conn.execute(text("SELECT COUNT(*) FROM projects"))
        count = result.scalar()
        print(f"  Total: {count} projet(s)")
        
        # Vérifier les owner_id
        result = await conn.execute(
            text("SELECT COUNT(*) FROM projects WHERE owner_id IS NOT NULL")
        )
        with_owner = result.scalar()
        print(f"  Avec propriétaire: {with_owner} projet(s)")
        
        print("\n✅ Configuration SaaS vérifiée avec succès!")


if __name__ == "__main__":
    asyncio.run(check_tables())
