#!/bin/bash
# Script pour importer le schéma Supabase dans PostgreSQL local

set -e  # Exit on error

echo "🔄 Importing Supabase schema to local PostgreSQL..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Import base schema
echo "📊 Importing base schema..."
docker-compose exec -T db psql -U genolens -d genolens_db < sql/supabase_schema_v2_final.sql

# Import enrichment pathways optimization
echo "🧬 Importing enrichment pathways schema..."
docker-compose exec -T db psql -U genolens -d genolens_db < sql/supabase_enrichment_pathways_schema.sql

echo "✅ Schema import complete!"
echo ""
echo "📝 Next steps:"
echo "1. Start all services: docker-compose up -d"
echo "2. Check API health: curl http://localhost:8001/health"
echo "3. View API logs: docker-compose logs api -f"
