#!/bin/bash
set -e

echo "🔄 Mise à jour GenoLens Backend..."

cd /home/dev/genolens_v2/backend

# Pull latest code
echo "📥 Récupération du code..."
git pull origin main

# Rebuild images
echo "🏗️  Rebuild des images..."
docker compose -f docker-compose.prod.yml build

# Run migrations
echo "🗄️  Migrations DB..."
docker compose -f docker-compose.prod.yml exec api alembic upgrade head

# Restart services (zero-downtime)
echo "♻️  Redémarrage des services..."
docker compose -f docker-compose.prod.yml up -d --no-deps --build api
sleep 10
docker compose -f docker-compose.prod.yml up -d --no-deps --build worker

echo "✅ Mise à jour terminée!"
docker compose -f docker-compose.prod.yml ps
