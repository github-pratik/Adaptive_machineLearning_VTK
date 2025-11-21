# Training PARIMA Models

This guide explains how to train machine learning models for PARIMA LOD (Level of Detail) prediction. The project supports both **RandomForest (PARIMA)** and **LSTM** models with comparison capabilities.

## Quick Start

### Option 1: Train and Compare Both Models (Recommended)

Train both RandomForest and LSTM models and compare their performance:

```bash
cd backend
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison
```

This will:
- Train RandomForest model (PARIMA)
- Train LSTM model with time-series cross-validation
- Generate comparison report
- Save both models to `model_comparison/` directory

### Option 2: Train Individual Model with Synthetic Data

For testing the training pipeline:

```bash
cd backend
python3 train_model.py --synthetic --samples 1000
```

This will:
- Generate 1000 synthetic training samples
- Train a RandomForest model (default)
- Save the model to `ml_models/PARIMA/model_checkpoint.pkl`

### Option 3: Train Individual Model with Real Data

Train a single model with collected data:

```bash
cd backend
python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
```

## Collecting Training Data

### Step-by-Step Process:

1. **Start the application**:
   - Start backend: `PORT=5001 npm run parima:backend`
   - Start frontend: `npm run start`

2. **Configure for data collection** in `config.json`:
   ```json
   {
     "parima": {
       "logging": {
         "enabled": true,
         "logFile": "./data/training_logs/parima_decisions_log.csv"
       }
     }
   }
   ```

3. **Load a VTP file** in the frontend

4. **Interact with the 3D model**:
   - Rotate the camera
   - Pan around
   - Zoom in/out
   - Let PARIMA collect decisions for 15-20 minutes

5. **Export logs**:
   - When buffer reaches 100 entries, CSV downloads automatically
   - Or click the "Copy Logs" button in the UI
   - Save CSV files to `data/training_logs/`

6. **Merge multiple CSV files** (if needed):
   ```bash
   cd backend
   python3 merge_training_data.py \
     --data-dir ../data/training_logs \
     --output parima_decisions_log.csv
   ```

7. **Train the models** with your collected data

## Model Types

### RandomForest (PARIMA)
- **Type**: Ensemble tree-based model
- **Best for**: Non-linear relationships, fast inference
- **Training**: Single train/test split (80/20)
- **Output**: `model_comparison/random_forest_model.pkl`

### LSTM
- **Type**: Recurrent neural network for time-series
- **Best for**: Temporal patterns, sequence data
- **Training**: Time-series cross-validation (5-fold)
- **Output**: `model_comparison/lstm_model.pkl`
- **Note**: Requires TensorFlow/Keras

## Model Comparison

### Train and Compare Both Models

```bash
cd backend
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison
```

### Generate Comparison Plots

```bash
cd backend
python3 generate_model_comparison_plots.py \
  --comparison-dir ../model_comparison \
  --output-dir ../model_comparison_plots \
  --csv-data ../data/training_logs/parima_decisions_log.csv
```

Plots generated:
- `1_accuracy_comparison.png` - Bar charts comparing accuracy
- `2_cv_comparison.png` - Cross-validation comparison
- `3_model_characteristics.png` - Model characteristics
- `4_accuracy_distribution.png` - Accuracy distribution
- `5_accuracy_boxplot.png` - Box-and-whisker plot
- `6_runtime_metrics_boxplot.png` - Runtime metrics comparison

### View Comparison Report

```bash
cat model_comparison/model_comparison_report.txt
```

## Command-Line Arguments

### compare_models.py

```bash
python3 compare_models.py [OPTIONS]
```

**Options:**
- `--data PATH` - Path to CSV file with training data (required)
- `--output-dir PATH` - Output directory for models and report (default: `./model_comparison`)

### train_model.py

```bash
python3 train_model.py [OPTIONS]
```

**Options:**
- `--data PATH` - Path to CSV file with training data
- `--synthetic` - Generate synthetic training data instead
- `--samples N` - Number of synthetic samples (default: 1000)
- `--model-type TYPE` - Model type: `random_forest`, `svm`, `logistic`, or `lstm` (default: `random_forest`)
- `--output PATH` - Output path for model file (default: `../ml_models/PARIMA/model_checkpoint.pkl`)

### Example Commands:

```bash
# Train and compare both models
python3 compare_models.py --data ../data/training_logs/parima_decisions_log.csv

# Train RandomForest only
python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv --model-type random_forest

# Train LSTM only
python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv --model-type lstm

# Train with synthetic data
python3 train_model.py --synthetic --samples 1000

# Train SVM model
python3 train_model.py --synthetic --model-type svm
```

## Feature Engineering

The model expects **35 features** total:

### Base Features (5):
1. `frustumCoverage` - Percentage of geometry visible (0-1)
2. `occlusionRatio` - Estimate of occluded geometry (0-1)
3. `meanVisibleDistance` - Average distance from camera (world units)
4. `deviceFPS` - Current frames per second
5. `deviceCPULoad` - Estimated CPU load (0-1)

### Trajectory Features (30):
- 10 history points × 3 features each
- Features: `viewportVelocity`, `viewportAcceleration`, `viewportAngularVelocity`
- If trajectory data isn't available in CSV, these are padded with zeros

## Data Collection for Both Models

### Collect PARIMA (RandomForest) Data

1. Update `config.json`:
   ```json
   {
     "modelPath": "./model_comparison/random_forest_model.pkl",
     "logging": {
       "logFile": "./data/training_logs/parima_decisions_log.csv"
     }
   }
   ```

2. Start backend and frontend
3. Collect data for 15-20 minutes
4. Save CSV to `data/training_logs/parima_decisions_log.csv`

### Collect LSTM Data

1. Update `config.json`:
   ```json
   {
     "modelPath": "./model_comparison/lstm_model.pkl",
     "logging": {
       "logFile": "./data/training_logs/lstm_decisions_log.csv"
     }
   }
   ```

2. Restart backend and frontend
3. Collect data for 15-20 minutes
4. Save CSV to `data/training_logs/lstm_decisions_log.csv`

## After Training

### 1. Verify Model Files

Check that model files were created:

```bash
ls -lh model_comparison/*.pkl
```

Should show:
- `random_forest_model.pkl`
- `lstm_model.pkl`

### 2. Update config.json

To use a specific model, update `config.json`:

**For RandomForest:**
```json
{
  "modelPath": "./model_comparison/random_forest_model.pkl"
}
```

**For LSTM:**
```json
{
  "modelPath": "./model_comparison/lstm_model.pkl"
}
```

### 3. Restart Backend

The backend automatically loads the model on startup:

```bash
PORT=5001 npm run parima:backend
```

### 4. Verify Model Loaded

Check the health endpoint:

```bash
curl http://localhost:5001/health
```

Should return:
```json
{"model_loaded": true, "model_path": "...", "status": "healthy"}
```

### 5. Test in Frontend

- Refresh your browser
- Check console for: `"PARIMA model loaded successfully"`
- Load a VTP file and watch PARIMA decisions
- Compare model predictions vs. fallback LOD

## Understanding Model Output

### compare_models.py Output

The comparison script generates:
- **Model files**: Both RandomForest and LSTM models
- **Comparison report**: `model_comparison_report.txt` with:
  - Training/test split information
  - Accuracy scores for both models
  - LSTM cross-validation results
  - Test set distribution analysis
  - Recommendations

### train_model.py Output

The training script prints:
- **Training/Test Split**: How data was divided
- **Feature Dimensions**: Should be 35
- **Model Accuracy**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score per LOD class
- **Feature Importance**: Top 5 most important features (RandomForest only)

### Example Output:

```
Training samples: 880
Test samples: 221
Feature dimensions: 35

Training random_forest model...

Model Accuracy: 0.8974

Classification Report:
              precision    recall  f1-score   support

        LOD 0       0.90      0.85      0.87        3
        LOD 2       0.82      0.88      0.85        80
        LOD 3       0.91      0.83      0.87        25
        LOD 4       0.95      0.96      0.95        120

    accuracy                           0.90       228
   macro avg       0.90      0.88      0.89       228
weighted avg       0.90      0.90      0.90       228

Top 5 Most Important Features:
  deviceFPS: 0.2845
  meanVisibleDistance: 0.1923
  frustumCoverage: 0.1567
  deviceCPULoad: 0.0989
  traj_0_vel: 0.0543

Model saved to: ../model_comparison/random_forest_model.pkl
```

## Troubleshooting

### "Missing required columns" Error

**Problem**: CSV file doesn't have required columns.

**Solution**: Ensure logging is enabled and you've collected data with all features. Check CSV has:
- `frustumCoverage`, `occlusionRatio`, `meanVisibleDistance`, `deviceFPS`, `deviceCPULoad`
- `decisionLOD` column

### "No valid training samples found" Error

**Problem**: All samples were filtered out (invalid LOD values or NaN).

**Solution**: 
- Check CSV has valid LOD values (0-5)
- Ensure no NaN or infinite values in feature columns
- Collect more data if needed

### LSTM Training Fails

**Problem**: TensorFlow not available or LSTM training errors.

**Solution**:
- Install TensorFlow: `pip3 install tensorflow`
- Check Python version: `python3 --version` (should be 3.7+)
- Verify enough data: LSTM needs at least 50 samples

### Model Accuracy is Low

**Possible Causes**:
- Not enough training data (collect more)
- Synthetic data doesn't match real patterns (use real data)
- Feature ranges don't match real usage (check feature distributions)

**Solutions**:
- Collect more real training data (aim for 500-1000+ samples)
- Try different model types
- Adjust model hyperparameters
- Check data quality and diversity

### Backend Still Shows "model_loaded: false"

**Check**:
1. Model file exists at correct path
2. File permissions are correct
3. Model file is valid pickle format
4. Check backend logs for error messages
5. Verify `config.json` model path is correct

### CSV Import Errors

**Problem**: pandas can't read CSV file.

**Solutions**:
- Ensure CSV is valid format
- Check for encoding issues (try UTF-8)
- Verify no malformed rows
- Check file path is correct

## Best Practices

### Data Collection
- Collect data across diverse scenarios (different camera movements, distances, device conditions)
- Aim for at least 500-1000 samples for reasonable performance
- Balance data across LOD classes (try to have samples for all LOD levels 0-5)
- Collect separate datasets for PARIMA and LSTM for fair comparison

### Model Selection
- Start with `random_forest` - good default, fast inference
- Try `lstm` for time-series patterns (but requires more data)
- Use `compare_models.py` to evaluate which performs better
- Consider inference speed vs. accuracy trade-offs

### Iterative Improvement
1. Train initial models with collected data
2. Deploy and collect more real data
3. Retrain with expanded dataset
4. Compare performance improvements
5. Fine-tune if needed
6. Repeat as more data becomes available

## Advanced Usage

### Hyperparameter Tuning

You can modify `train_model.py` to add hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

# Example for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
```

### Cross-Validation

LSTM uses time-series cross-validation automatically. For RandomForest:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## Support

For issues or questions:
1. Check backend logs for error messages
2. Verify all dependencies are installed: `pip3 install -r requirements.txt`
3. Ensure CSV format matches expected structure
4. Test with synthetic data first to verify pipeline works
5. See `MODEL_COMPARISON_GUIDE.md` for detailed comparison instructions
