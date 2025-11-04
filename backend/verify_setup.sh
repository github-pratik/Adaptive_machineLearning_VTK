#!/bin/bash
# Verification script for PARIMA setup
# Checks all components are in place and working

echo "=========================================="
echo "PARIMA Setup Verification"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# Check 1: Model file exists
echo -e "\n[1/6] Checking model file..."
MODEL_PATH="ml_models/PARIMA/model_checkpoint.pkl"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    echo -e "${GREEN}✅ Model file exists: $MODEL_PATH ($SIZE)${NC}"
else
    echo -e "${RED}❌ Model file not found: $MODEL_PATH${NC}"
    echo "   Run: python3 backend/train_model.py --synthetic"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Python dependencies
echo -e "\n[2/6] Checking Python dependencies..."
if python3 -c "import flask, numpy, sklearn, pandas" 2>/dev/null; then
    echo -e "${GREEN}✅ All Python dependencies installed${NC}"
else
    echo -e "${RED}❌ Missing Python dependencies${NC}"
    echo "   Run: pip3 install -r backend/requirements.txt"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Model can be loaded
echo -e "\n[3/6] Testing model loading..."
if python3 -c "import pickle; pickle.load(open('$MODEL_PATH', 'rb'))" 2>/dev/null; then
    echo -e "${GREEN}✅ Model can be loaded${NC}"
else
    echo -e "${RED}❌ Model file is corrupted or incompatible${NC}"
    echo "   Run: python3 backend/train_model.py --synthetic"
    ERRORS=$((ERRORS + 1))
fi

# Check 4: Backend can import modules
echo -e "\n[4/6] Testing backend imports..."
cd backend 2>/dev/null
if python3 -c "from parima_api import app, load_model" 2>/dev/null; then
    echo -e "${GREEN}✅ Backend modules import successfully${NC}"
    cd ..
else
    echo -e "${RED}❌ Backend imports failed${NC}"
    cd ..
    ERRORS=$((ERRORS + 1))
fi

# Check 5: Config file exists
echo -e "\n[5/6] Checking config file..."
if [ -f "config.json" ]; then
    echo -e "${GREEN}✅ config.json exists${NC}"
else
    echo -e "${YELLOW}⚠️  config.json not found (optional)${NC}"
fi

# Check 6: Node dependencies
echo -e "\n[6/6] Checking Node.js dependencies..."
if [ -d "node_modules" ]; then
    echo -e "${GREEN}✅ Node modules installed${NC}"
else
    echo -e "${YELLOW}⚠️  Node modules not found${NC}"
    echo "   Run: npm install"
fi

# Summary
echo -e "\n=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo -e "\nNext steps:"
    echo "1. Start backend: PORT=5001 python3 backend/parima_api.py"
    echo "2. Start frontend: npm run start"
    echo "3. Test API: python3 backend/test_api.py"
    exit 0
else
    echo -e "${RED}❌ Found $ERRORS error(s)${NC}"
    echo "Please fix the errors above before proceeding"
    exit 1
fi

