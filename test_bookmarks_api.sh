#!/bin/bash

# Test script for Gene Bookmarks API
# Usage: ./test_bookmarks_api.sh

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gene Bookmarks API Test Script ===${NC}\n"

# Configuration
API_URL="http://localhost:8000/api/v1"
PROJECT_ID="your-project-id-here"
TOKEN="your-token-here"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed. Please install it first.${NC}"
    echo "  macOS: brew install jq"
    echo "  Ubuntu: sudo apt-get install jq"
    exit 1
fi

# Function to make API calls
call_api() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -z "$data" ]; then
        curl -s -X "$method" \
            "${API_URL}${endpoint}" \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Content-Type: application/json"
    else
        curl -s -X "$method" \
            "${API_URL}${endpoint}" \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$data"
    fi
}

# Test 1: Create a bookmark
echo -e "${BLUE}Test 1: Create bookmark for TP53${NC}"
BOOKMARK_RESPONSE=$(call_api POST "/projects/${PROJECT_ID}/bookmarks" '{
  "gene_symbol": "TP53",
  "notes": "Important tumor suppressor gene",
  "tags": ["cancer", "important"],
  "color": "#ef4444",
  "is_favorite": true
}')

if echo "$BOOKMARK_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
    BOOKMARK_ID=$(echo "$BOOKMARK_RESPONSE" | jq -r '.id')
    echo -e "${GREEN}✓ Bookmark created successfully${NC}"
    echo "  ID: $BOOKMARK_ID"
    echo "  Gene: $(echo "$BOOKMARK_RESPONSE" | jq -r '.gene_symbol')"
else
    echo -e "${RED}✗ Failed to create bookmark${NC}"
    echo "$BOOKMARK_RESPONSE" | jq .
    exit 1
fi

# Test 2: Get all bookmarks
echo -e "\n${BLUE}Test 2: Get all bookmarks${NC}"
BOOKMARKS=$(call_api GET "/projects/${PROJECT_ID}/bookmarks")
BOOKMARK_COUNT=$(echo "$BOOKMARKS" | jq 'length')
echo -e "${GREEN}✓ Found $BOOKMARK_COUNT bookmark(s)${NC}"

# Test 3: Check if gene is bookmarked
echo -e "\n${BLUE}Test 3: Check if TP53 is bookmarked${NC}"
CHECK_RESPONSE=$(call_api GET "/projects/${PROJECT_ID}/bookmarks/check/TP53")
IS_BOOKMARKED=$(echo "$CHECK_RESPONSE" | jq -r '.is_bookmarked')
if [ "$IS_BOOKMARKED" = "true" ]; then
    echo -e "${GREEN}✓ TP53 is bookmarked${NC}"
else
    echo -e "${RED}✗ TP53 is not bookmarked${NC}"
fi

# Test 4: Update bookmark
echo -e "\n${BLUE}Test 4: Update bookmark notes${NC}"
UPDATE_RESPONSE=$(call_api PUT "/bookmarks/${BOOKMARK_ID}" '{
  "notes": "Updated: Critical tumor suppressor, mutated in many cancers",
  "tags": ["cancer", "important", "p53-pathway"]
}')
if echo "$UPDATE_RESPONSE" | jq -e '.notes' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Bookmark updated successfully${NC}"
    echo "  Notes: $(echo "$UPDATE_RESPONSE" | jq -r '.notes' | head -c 50)..."
else
    echo -e "${RED}✗ Failed to update bookmark${NC}"
fi

# Test 5: Create multiple bookmarks (batch)
echo -e "\n${BLUE}Test 5: Create batch bookmarks${NC}"
BATCH_RESPONSE=$(call_api POST "/projects/${PROJECT_ID}/bookmarks/batch" '{
  "gene_symbols": ["BRCA1", "BRCA2", "MYC", "KRAS"],
  "tags": ["batch-test", "cancer"],
  "notes": "Batch created test genes",
  "is_favorite": true
}')
CREATED_COUNT=$(echo "$BATCH_RESPONSE" | jq -r '.created')
SKIPPED_COUNT=$(echo "$BATCH_RESPONSE" | jq -r '.skipped')
echo -e "${GREEN}✓ Batch create completed${NC}"
echo "  Created: $CREATED_COUNT"
echo "  Skipped: $SKIPPED_COUNT"

# Test 6: Create a gene list
echo -e "\n${BLUE}Test 6: Create gene list${NC}"
LIST_RESPONSE=$(call_api POST "/projects/${PROJECT_ID}/gene-lists" '{
  "name": "Cancer Markers",
  "description": "Important cancer-related genes",
  "genes": ["TP53", "BRCA1", "BRCA2", "MYC", "KRAS", "EGFR"],
  "color": "#3b82f6",
  "is_public": true,
  "tags": ["cancer", "markers"]
}')
if echo "$LIST_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
    LIST_ID=$(echo "$LIST_RESPONSE" | jq -r '.id')
    GENE_COUNT=$(echo "$LIST_RESPONSE" | jq -r '.gene_count')
    echo -e "${GREEN}✓ Gene list created successfully${NC}"
    echo "  ID: $LIST_ID"
    echo "  Name: $(echo "$LIST_RESPONSE" | jq -r '.name')"
    echo "  Genes: $GENE_COUNT"
else
    echo -e "${RED}✗ Failed to create gene list${NC}"
    echo "$LIST_RESPONSE" | jq .
    exit 1
fi

# Test 7: Get all gene lists
echo -e "\n${BLUE}Test 7: Get all gene lists${NC}"
LISTS=$(call_api GET "/projects/${PROJECT_ID}/gene-lists?include_public=true")
LIST_COUNT=$(echo "$LISTS" | jq 'length')
echo -e "${GREEN}✓ Found $LIST_COUNT gene list(s)${NC}"

# Test 8: Add genes to list
echo -e "\n${BLUE}Test 8: Add genes to list${NC}"
ADD_RESPONSE=$(call_api POST "/gene-lists/${LIST_ID}/add-genes" '{
  "genes": ["AKT1", "PTEN"]
}')
NEW_COUNT=$(echo "$ADD_RESPONSE" | jq -r '.gene_count')
echo -e "${GREEN}✓ Genes added to list${NC}"
echo "  New count: $NEW_COUNT"

# Test 9: Remove genes from list
echo -e "\n${BLUE}Test 9: Remove genes from list${NC}"
REMOVE_RESPONSE=$(call_api POST "/gene-lists/${LIST_ID}/remove-genes" '{
  "genes": ["AKT1"]
}')
FINAL_COUNT=$(echo "$REMOVE_RESPONSE" | jq -r '.gene_count')
echo -e "${GREEN}✓ Genes removed from list${NC}"
echo "  Final count: $FINAL_COUNT"

# Test 10: Delete gene list
echo -e "\n${BLUE}Test 10: Delete gene list${NC}"
DELETE_LIST_RESPONSE=$(call_api DELETE "/gene-lists/${LIST_ID}")
if [ -z "$DELETE_LIST_RESPONSE" ]; then
    echo -e "${GREEN}✓ Gene list deleted successfully${NC}"
else
    echo -e "${RED}✗ Failed to delete gene list${NC}"
fi

# Test 11: Delete bookmark
echo -e "\n${BLUE}Test 11: Delete bookmark${NC}"
DELETE_RESPONSE=$(call_api DELETE "/bookmarks/${BOOKMARK_ID}")
if [ -z "$DELETE_RESPONSE" ]; then
    echo -e "${GREEN}✓ Bookmark deleted successfully${NC}"
else
    echo -e "${RED}✗ Failed to delete bookmark${NC}"
fi

# Clean up batch bookmarks
echo -e "\n${BLUE}Cleanup: Deleting batch bookmarks${NC}"
BATCH_BOOKMARKS=$(call_api GET "/projects/${PROJECT_ID}/bookmarks")
echo "$BATCH_BOOKMARKS" | jq -r '.[].id' | while read -r id; do
    call_api DELETE "/bookmarks/$id" > /dev/null 2>&1
done
echo -e "${GREEN}✓ Cleanup completed${NC}"

echo -e "\n${GREEN}=== All tests completed! ===${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo "1. Test the UI by opening your project in the browser"
echo "2. Click on the ⭐ icon next to genes in DEG tables"
echo "3. Open the Bookmark Manager to see your bookmarks"
echo "4. Create custom gene lists via the Gene List Manager"
