#!/bin/bash
# Helper script to start PARIMA backend with verification

echo "Starting PARIMA Backend Server..."

# Check if model file exists
MODEL_PATH="../ml_models/PARIMA/model_checkpoint.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Warning: Model file not found at $MODEL_PATH"
    echo "   Training a new model with synthetic data..."
    python3 train_model.py --synthetic --samples 1000 --output "$MODEL_PATH"
    if [ $? -eq 0 ]; then
        echo "✅ Model trained successfully"
    else
        echo "⚠️  Model training failed, but continuing..."
    fi
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask, numpy, sklearn, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install -r requirements.txt
fi

# Set port
PORT=${PORT:-5001}

echo ""
echo "Starting server on port $PORT..."
echo "Press Ctrl+C to stop"
echo ""

PORT=$PORT python3 parima_api.py

