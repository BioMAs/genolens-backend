#!/bin/bash

# Script de démarrage pour tester GeneLens v2 avec optimisations

set -e

echo "🚀 GeneLens v2 - Quick Start Testing"
echo "===================================="
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Erreur: Ce script doit être exécuté depuis backend/"
    exit 1
fi

echo "📦 Étape 1/5: Démarrage des services Docker..."
docker-compose up -d postgres redis

echo "⏳ Attente que PostgreSQL soit prêt..."
sleep 5

until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "   Attente PostgreSQL..."
    sleep 2
done
echo "   ✅ PostgreSQL prêt"

echo ""
echo "🗄️  Étape 2/5: Application des migrations..."
alembic upgrade head

echo ""
echo "🔧 Étape 3/5: Démarrage de l'API..."
docker-compose up -d api

echo "⏳ Attente que l'API soit prête..."
sleep 5

until curl -s http://localhost:8001/health > /dev/null 2>&1; do
    echo "   Attente API..."
    sleep 2
done
echo "   ✅ API prête"

echo ""
echo "✅ Étape 4/5: Vérification des services..."
echo ""

# Vérifier PostgreSQL
PG_STATUS=$(docker-compose exec -T postgres pg_isready -U postgres | grep -c "accepting connections" || echo "0")
if [ "$PG_STATUS" -gt 0 ]; then
    echo "   ✅ PostgreSQL: OK"
else
    echo "   ❌ PostgreSQL: ERREUR"
fi

# Vérifier Redis
REDIS_STATUS=$(docker-compose exec -T redis redis-cli ping | grep -c "PONG" || echo "0")
if [ "$REDIS_STATUS" -gt 0 ]; then
    echo "   ✅ Redis: OK"
else
    echo "   ❌ Redis: ERREUR"
fi

# Vérifier API
API_STATUS=$(curl -s http://localhost:$API_PORT/health | grep -c "healthy" || echo "0")
if [ "$API_STATUS" -gt 0 ]; then
    echo "   ✅ API: OK"
else
    echo "   ❌ API: ERREUR"
fi

echo ""
echo "📊 Étape 5/5: Services disponibles"
echo ""
echo "   🌐 API Backend:  http://localhost:8001"
echo "   📖 API Docs:     http://localhost:8001/docs"
echo "   🐘 PostgreSQL:   localhost:5432"
echo "   🔴 Redis:        localhost:6379"
echo ""
echo "🧪 Pour tester les optimisations:"
echo ""
echo "   # Tests automatisés"
echo "   pytest tests/test_performance.py -v"
echo ""
echo "   # Monitoring"
echo "   curl http://localhost:8001/api/datasets/admin/performance-stats | jq"
echo ""
echo "   # Voir les logs"
echo "   docker-compose logs -f api"
echo ""
echo "✅ Tout est prêt! Bon testing 🎉"
echo ""
echo "Pour arrêter: docker-compose down"
