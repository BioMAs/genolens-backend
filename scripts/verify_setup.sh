#!/bin/bash

# GenoLens Next - Setup Verification Script
# This script checks if all required services are running and configured correctly

echo "🔍 GenoLens Next - Setup Verification"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
echo "1️⃣  Checking Docker..."
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker is running"
else
    echo -e "${RED}✗${NC} Docker is not running"
    exit 1
fi

# Check if .env file exists
echo ""
echo "2️⃣  Checking environment configuration..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"
else
    echo -e "${RED}✗${NC} .env file not found"
    echo -e "${YELLOW}→${NC} Run: cp .env.example .env"
    exit 1
fi

# Check if docker-compose services are running
echo ""
echo "3️⃣  Checking Docker Compose services..."

services=("postgres" "redis" "api" "worker")
all_running=true

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✓${NC} $service is running"
    else
        echo -e "${RED}✗${NC} $service is not running"
        all_running=false
    fi
done

if [ "$all_running" = false ]; then
    echo -e "${YELLOW}→${NC} Run: docker-compose up -d"
    exit 1
fi

# Check API health
echo ""
echo "4️⃣  Checking API health..."
sleep 2  # Give API time to start
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} API is responding"
    curl -s http://localhost:8001/health | python3 -m json.tool
else
    echo -e "${RED}✗${NC} API is not responding"
    echo -e "${YELLOW}→${NC} Check logs: docker-compose logs api"
    exit 1
fi

# Check Redis connection
echo ""
echo "5️⃣  Checking Redis connection..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Redis is responding"
else
    echo -e "${RED}✗${NC} Redis is not responding"
    exit 1
fi

# Check database connection
echo ""
echo "6️⃣  Checking database connection..."
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Database is ready"
else
    echo -e "${RED}✗${NC} Database is not ready"
    exit 1
fi

# Summary
echo ""
echo "======================================"
echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo "Next steps:"
echo "1. Run migrations: make migrate"
echo "2. Visit API docs: http://localhost:8000/docs"
echo "3. Monitor tasks: http://localhost:5555 (Flower)"
echo ""
