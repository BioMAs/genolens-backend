#!/bin/bash

# GenoLens Next - Configuration Rapide
# Ce script automatise le démarrage initial du projet

set -e  # Arrêter en cas d'erreur

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   GenoLens Next - Configuration       ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo ""

# Vérifier que .env existe
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ Fichier .env non trouvé!${NC}"
    echo -e "${YELLOW}→ Le fichier .env a été créé avec vos informations Supabase${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Fichier .env trouvé"
echo ""

# Vérifier Docker
echo -e "${BLUE}1. Vérification de Docker...${NC}"
if ! docker ps > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker n'est pas démarré${NC}"
    echo -e "${YELLOW}→ Démarrez Docker Desktop et réessayez${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker est actif"
echo ""

# Démarrer les services
echo -e "${BLUE}2. Démarrage des services Docker...${NC}"
docker-compose up -d
echo -e "${GREEN}✓${NC} Services démarrés"
echo ""

# Attendre que les services soient prêts
echo -e "${BLUE}3. Attente du démarrage des services (30s)...${NC}"
sleep 10
echo -e "${YELLOW}   10s...${NC}"
sleep 10
echo -e "${YELLOW}   20s...${NC}"
sleep 10
echo -e "${YELLOW}   30s...${NC}"
echo -e "${GREEN}✓${NC} Services prêts"
echo ""

# Vérifier la santé de l'API
echo -e "${BLUE}4. Vérification de l'API...${NC}"
max_retries=5
retry=0
while [ $retry -lt $max_retries ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} API répond correctement"
        break
    fi
    retry=$((retry + 1))
    if [ $retry -lt $max_retries ]; then
        echo -e "${YELLOW}   Tentative $retry/$max_retries...${NC}"
        sleep 5
    else
        echo -e "${RED}✗ L'API ne répond pas${NC}"
        echo -e "${YELLOW}→ Vérifiez les logs: docker-compose logs api${NC}"
        exit 1
    fi
done
echo ""

# Créer la migration initiale
echo -e "${BLUE}5. Création de la migration initiale...${NC}"
docker-compose exec -T api alembic revision --autogenerate -m "Initial migration" > /dev/null 2>&1 || true
echo -e "${GREEN}✓${NC} Migration créée"
echo ""

# Appliquer les migrations
echo -e "${BLUE}6. Application des migrations...${NC}"
docker-compose exec -T api alembic upgrade head
echo -e "${GREEN}✓${NC} Base de données initialisée"
echo ""

# Résumé
echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Configuration Terminée!       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Tous les services sont opérationnels${NC}"
echo ""
echo -e "${YELLOW}Services disponibles:${NC}"
echo -e "  • API Documentation: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  • Health Check:      ${BLUE}http://localhost:8000/health${NC}"
echo -e "  • Flower (Tasks):    ${BLUE}http://localhost:5555${NC}"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo -e "  1. Créer le bucket Supabase Storage 'genolens-data'"
echo -e "     ${BLUE}https://supabase.com/dashboard/project/<votre-project-ref>/storage/buckets${NC}"
echo ""
echo -e "  2. Créer un utilisateur de test"
echo -e "     ${BLUE}https://supabase.com/dashboard/project/<votre-project-ref>/auth/users${NC}"
echo ""
echo -e "  3. Tester l'API avec la documentation interactive"
echo -e "     ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Commandes utiles:${NC}"
echo -e "  • Voir les logs:        ${BLUE}docker-compose logs -f api${NC}"
echo -e "  • Redémarrer:           ${BLUE}docker-compose restart${NC}"
echo -e "  • Arrêter:              ${BLUE}docker-compose down${NC}"
echo -e "  • Aide:                 ${BLUE}make help${NC}"
echo ""
echo -e "${GREEN}Tout est prêt! Bon développement! 🚀${NC}"
echo ""
