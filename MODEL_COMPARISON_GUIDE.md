# PARIMA vs LSTM Model Comparison Guide

This guide explains how to compare PARIMA (RandomForest) and LSTM models for adaptive streaming.

## Overview

The project now supports two model types:
- **RandomForest (PARIMA)**: Traditional machine learning, fast inference, smaller model
- **LSTM**: Deep learning with temporal sequence understanding, more complex

## Prerequisites

1. **Install TensorFlow** (required for LSTM):
   ```bash
   cd backend
   pip3 install tensorflow>=2.10.0
   ```

2. **Ensure you have training data**:
   - Collect data by running the application with PARIMA enabled
   - Data is logged to: `data/training_logs/parima_decisions_log.csv`
   - You need at least 30 samples for LSTM (more is better)

## Step 1: Train Both Models

Run the comparison script to train both models:

```bash
cd backend
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison
```

This will:
- Train a RandomForest model (PARIMA)
- Train an LSTM model
- Generate a comparison report
- Save both models to `model_comparison/` directory

### Output Files

- `model_comparison/random_forest_model.pkl` - RandomForest model
- `model_comparison/lstm_model.pkl` - LSTM model
- `model_comparison/model_comparison_report.txt` - Detailed comparison report

## Step 2: Review Comparison Report

Open the comparison report:

```bash
cat model_comparison/model_comparison_report.txt
```

The report includes:
- Training data statistics
- Model accuracy for both models
- Accuracy difference and percentage improvement
- Recommendations based on results

## Step 3: Test Each Model

### Test with RandomForest (PARIMA)

1. Update `config.json`:
   ```json
   {
     "parima": {
       "enabled": true,
       "modelPath": "./model_comparison/random_forest_model.pkl",
       "training": {
         "modelType": "random_forest"
       }
     }
   }
   ```

2. Restart backend:
   ```bash
   PORT=5001 npm run parima:backend
   ```

3. Run the application and collect performance metrics

### Test with LSTM

1. Update `config.json`:
   ```json
   {
     "parima": {
       "enabled": true,
       "modelPath": "./model_comparison/lstm_model.pkl",
       "training": {
         "modelType": "lstm"
       }
     }
   }
   ```

2. Restart backend:
   ```bash
   PORT=5001 npm run parima:backend
   ```

3. Run the application and collect performance metrics

## Step 4: Compare Performance

Collect data for both models and compare:

1. **Accuracy**: Test set accuracy (from comparison report)
2. **Inference Latency**: API response time
3. **Model Size**: File size on disk
4. **Training Time**: Time to train each model
5. **Resource Usage**: Memory/CPU during inference
6. **Prediction Distribution**: LOD usage patterns

### Metrics to Track

- **FPS**: Frame rate during streaming
- **GPU Load**: GPU utilization
- **Latency**: API response time
- **Memory**: Peak memory usage
- **LOD Distribution**: Which LOD levels are used most

## Model Characteristics

### RandomForest (PARIMA)

**Advantages:**
- ✅ Fast inference (~1-5ms)
- ✅ Small model size (~1-5MB)
- ✅ Simple deployment
- ✅ No sequence history needed
- ✅ Works well with limited data

**Disadvantages:**
- ❌ No temporal understanding
- ❌ Treats each prediction independently

### LSTM

**Advantages:**
- ✅ Temporal sequence understanding
- ✅ Can learn patterns over time
- ✅ Potentially higher accuracy for time-series data
- ✅ Research value for deep learning comparison

**Disadvantages:**
- ❌ Slower inference (~10-50ms)
- ❌ Larger model size (~10-50MB)
- ❌ Requires sequence history (10 previous samples)
- ❌ Needs more training data (30+ samples minimum)
- ❌ More complex deployment

## Troubleshooting

### LSTM Training Fails

**Error**: "TensorFlow not available"
- **Solution**: Install TensorFlow: `pip3 install tensorflow>=2.10.0`

**Error**: "Need at least 30 samples"
- **Solution**: Collect more training data by running the application longer

**Error**: Memory issues during training
- **Solution**: Reduce batch size in `train_model.py` (line 306: `batch_size=32` → `batch_size=16`)

### LSTM Inference Issues

**Error**: "Model not loaded" or prediction fails
- **Solution**: Ensure `modelPath` in `config.json` points to the correct LSTM model file

**Issue**: Predictions seem random initially
- **Explanation**: LSTM needs 10 samples of history before making accurate predictions. The first 10 predictions use repeated current features.

## Manual Training

You can also train models individually:

### Train RandomForest:
```bash
cd backend
python3 train_model.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --model-type random_forest \
  --output ../model_comparison/random_forest_model.pkl
```

### Train LSTM:
```bash
cd backend
python3 train_model.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --model-type lstm \
  --output ../model_comparison/lstm_model.pkl
```

## Research Questions

When comparing models, consider:

1. **Which model provides better FPS stability?**
   - Use FPS coefficient of variation from logs

2. **Which model uses resources more efficiently?**
   - Compare FPS per GPU% and FPS per Memory MB

3. **Which model adapts better to changing conditions?**
   - Analyze LOD change frequency and appropriateness

4. **Is the accuracy improvement worth the complexity?**
   - Compare accuracy vs inference latency trade-off

## Next Steps

1. Train both models on your collected data
2. Review the comparison report
3. Test each model in the application
4. Collect performance metrics for both
5. Generate comparison plots using `generate_comparison_plots.py`
6. Document findings for your research

## Example Workflow

```bash
# 1. Collect training data (run application)
# 2. Train and compare models
cd backend
python3 compare_models.py --data ../data/training_logs/parima_decisions_log.csv

# 3. Review report
cat ../model_comparison/model_comparison_report.txt

# 4. Test RandomForest
# (Update config.json, restart backend, run app, collect data)

# 5. Test LSTM
# (Update config.json, restart backend, run app, collect data)

# 6. Generate comparison plots
python3 generate_comparison_plots.py \
  --parima-data ../data/training_logs/parima_decisions_log.csv \
  --baseline-data ../data/training_logs/baseline_no_parima.csv \
  --output-dir ../comparison_plots
```

