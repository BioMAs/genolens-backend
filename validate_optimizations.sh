#!/bin/bash

# Script de validation rapide des optimisations GeneLens v2

API_URL="http://localhost:8001"
API_BASE="$API_URL/api/v1"
DATASET_ID="${1:-}"

echo "🔍 Validation des Optimisations GeneLens v2"
echo "==========================================="
echo ""

if [ -z "$DATASET_ID" ]; then
    echo "💡 Usage: ./validate_optimizations.sh <dataset_id>"
    echo ""
    echo "   Pour trouver un dataset_id:"
    echo "   docker-compose exec postgres psql -U postgres -d genolens -c 'SELECT id, name FROM datasets LIMIT 5;'"
    echo ""
    exit 1
fi

echo "🎯 Testing dataset: $DATASET_ID"
echo ""

# ── Test 1: Santé API ─────────────────────────────────────────────────────────
echo "1️⃣  Test: API Health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ API accessible (HTTP 200)"
else
    echo "   ❌ API non accessible (HTTP $HTTP_CODE)"
    exit 1
fi

# ── Test 2: Endpoint /columns ─────────────────────────────────────────────────
echo ""
echo "2️⃣  Test: Endpoint /columns (<50ms attendu)..."
START=$(date +%s%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/datasets/$DATASET_ID/columns")
END=$(date +%s%N)
COLUMNS_DURATION=$(( (END - START) / 1000000 ))
echo "   ⏱️  Durée: ${COLUMNS_DURATION}ms  HTTP: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ] && [ $COLUMNS_DURATION -lt 50 ]; then
    echo "   ✅ Performance OK"
elif [ "$HTTP_CODE" = "403" ]; then
    echo "   ℹ️  Auth requise (normal si Supabase JWT non configuré)"
else
    echo "   ⚠️  Plus lent qu'attendu (${COLUMNS_DURATION}ms > 50ms)"
fi

# ── Test 3: Endpoint /stats ───────────────────────────────────────────────────
echo ""
echo "3️⃣  Test: Endpoint /stats (<200ms attendu)..."
START=$(date +%s%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/datasets/$DATASET_ID/stats")
END=$(date +%s%N)
STATS_DURATION=$(( (END - START) / 1000000 ))
echo "   ⏱️  Durée: ${STATS_DURATION}ms  HTTP: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ] && [ $STATS_DURATION -lt 200 ]; then
    echo "   ✅ Performance OK"
elif [ "$HTTP_CODE" = "403" ]; then
    echo "   ℹ️  Auth requise (normal si Supabase JWT non configuré)"
else
    echo "   ⚠️  Plus lent qu'attendu (${STATS_DURATION}ms > 200ms)"
fi

# ── Test 4: Endpoint /genes/list ──────────────────────────────────────────────
echo ""
echo "4️⃣  Test: Endpoint /genes/list (<100ms attendu)..."
START=$(date +%s%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/datasets/$DATASET_ID/genes/list")
END=$(date +%s%N)
GENES_DURATION=$(( (END - START) / 1000000 ))
echo "   ⏱️  Durée: ${GENES_DURATION}ms  HTTP: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ] && [ $GENES_DURATION -lt 100 ]; then
    echo "   ✅ Performance OK"
elif [ "$HTTP_CODE" = "403" ]; then
    echo "   ℹ️  Auth requise (normal si Supabase JWT non configuré)"
else
    echo "   ⚠️  Plus lent qu'attendu (${GENES_DURATION}ms > 100ms)"
fi

# ── Test 5: Cache clustering ──────────────────────────────────────────────────
echo ""
echo "5️⃣  Test: Cache Effectiveness (clustering)..."
echo "   Premier appel (cache MISS)..."
START=$(date +%s%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/datasets/$DATASET_ID/cluster" \
    -H "Content-Type: application/json" \
    -d '{"top_n_genes": 100, "cluster_rows": true, "cluster_cols": true}')
END=$(date +%s%N)
MISS_DURATION=$(( (END - START) / 1000000 ))
echo "   ⏱️  Cache MISS: ${MISS_DURATION}ms  HTTP: $HTTP_CODE"

if [ "$HTTP_CODE" = "200" ]; then
    sleep 1

    echo "   Deuxième appel (cache HIT attendu)..."
    START=$(date +%s%N)
    curl -s -o /dev/null -X POST "$API_BASE/datasets/$DATASET_ID/cluster" \
        -H "Content-Type: application/json" \
        -d '{"top_n_genes": 100, "cluster_rows": true, "cluster_cols": true}'
    END=$(date +%s%N)
    HIT_DURATION=$(( (END - START) / 1000000 ))
    echo "   ⏱️  Cache HIT:  ${HIT_DURATION}ms"

    if [ $HIT_DURATION -gt 0 ]; then
        SPEEDUP=$(( MISS_DURATION / HIT_DURATION ))
        if [ $SPEEDUP -gt 5 ]; then
            echo "   ✅ Cache efficace (${SPEEDUP}x plus rapide)"
        else
            echo "   ⚠️  Cache peu efficace (${SPEEDUP}x plus rapide, <5x attendu)"
        fi
    fi
else
    echo "   ℹ️  Test cache ignoré (auth requise ou endpoint non disponible)"
    HIT_DURATION=0
    SPEEDUP=0
fi

# ── Test 6: Monitoring endpoints ──────────────────────────────────────────────
echo ""
echo "6️⃣  Test: Monitoring Endpoints (admin)..."
PERF_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/datasets/admin/performance-stats")
CACHE_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/datasets/admin/cache-stats")

if [ "$PERF_CODE" = "200" ] && [ "$CACHE_CODE" = "200" ]; then
    echo "   ✅ Endpoints monitoring accessibles"
elif [ "$PERF_CODE" = "403" ]; then
    echo "   ℹ️  Endpoints monitoring existent — auth admin requise (HTTP 403)"
else
    echo "   ❌ Endpoints monitoring manquants (HTTP perf=$PERF_CODE cache=$CACHE_CODE)"
fi

# ── Test 7: Vérifications DB ──────────────────────────────────────────────────
echo ""
echo "7️⃣  Test: Database Optimizations..."

echo "   Vérification schema tables..."
TABLES_OK=$(docker-compose exec -T postgres psql -U postgres -d genolens -tAc \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('cached_computations','project_comments','project_activity_log','sample_correlations','go_terms','go_annotations')" 2>/dev/null || echo "0")
if [ "$TABLES_OK" = "6" ]; then
    echo "   ✅ Toutes les tables attendues présentes (6/6)"
else
    echo "   ⚠️  Tables présentes: $TABLES_OK/6"
fi

echo "   Vérification colonne deg_up_count sur datasets..."
STATS_COL=$(docker-compose exec -T postgres psql -U postgres -d genolens -tAc \
    "SELECT 1 FROM information_schema.columns WHERE table_name='datasets' AND column_name='deg_up_count'" 2>/dev/null | tr -d ' ')
if [ "$STATS_COL" = "1" ]; then
    echo "   ✅ Colonnes stats présentes (deg_up_count)"
else
    echo "   ❌ Colonnes stats manquantes (migration performance_optimizations)"
fi

echo "   Vérification migration Alembic à jour..."
ALEMBIC_HEAD=$(docker-compose exec -T api alembic current 2>/dev/null | grep "(head)" | awk '{print $1}')
if [ -n "$ALEMBIC_HEAD" ]; then
    echo "   ✅ DB à jour — revision: $ALEMBIC_HEAD"
else
    echo "   ⚠️  DB possiblement en retard (vérifier: alembic upgrade head)"
fi

# ── Résumé ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "📊 RÉSUMÉ"
echo "============================================"
echo ""
echo "Endpoints (temps de réponse):"
echo "  • /columns:     ${COLUMNS_DURATION}ms  (cible: <50ms)"
echo "  • /stats:       ${STATS_DURATION}ms  (cible: <200ms)"
echo "  • /genes/list:  ${GENES_DURATION}ms  (cible: <100ms)"
echo ""
if [ $SPEEDUP -gt 0 ]; then
    echo "Cache clustering:"
    echo "  • MISS: ${MISS_DURATION}ms"
    echo "  • HIT:  ${HIT_DURATION}ms"
    echo "  • Speedup: ${SPEEDUP}x (cible: >5x)"
    echo ""
fi
echo "Pour détails monitoring (avec JWT admin) :"
echo "  curl $API_BASE/datasets/admin/performance-stats | jq"
echo "  curl $API_BASE/datasets/admin/cache-stats | jq"
echo ""
echo "✅ Validation terminée!"
