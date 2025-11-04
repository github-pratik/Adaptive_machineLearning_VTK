# Training PARIMA Model

This guide explains how to train a machine learning model for PARIMA LOD (Level of Detail) prediction.

## Quick Start

### Option 1: Train with Synthetic Data (For Testing)

This is the fastest way to get started and test the training pipeline:

```bash
cd backend
python3 train_model.py --synthetic --samples 1000
```

This will:
- Generate 1000 synthetic training samples
- Train a RandomForest model
- Save the model to `ml_models/PARIMA/model_checkpoint.pkl`

### Option 2: Train with Real Collected Data

For best results, train with real data collected from your application:

1. **Collect data** by running the frontend with logging enabled
2. **Export logs** (CSV file will be downloaded from browser via "Copy Logs" button)
3. **Train model**:
   ```bash
   cd backend
   python3 train_model.py --data path/to/parima_decisions_log.csv
   ```

## Collecting Training Data

### Step-by-Step Process:

1. **Start the application**:
   - Start backend: `PORT=5001 python3 backend/parima_api.py`
   - Start frontend: `npm run start`

2. **Enable logging** in `config.json`:
   ```json
   {
     "parima": {
       "logging": {
         "enabled": false, //true to download csv, and false to automate parima
         "logFile": "parima_decisions_log.csv"
       }
     }
   }
   ```

3. **Load a VTP file** in the frontend

4. **Interact with the 3D model**:
   - Rotate the camera
   - Pan around
   - Zoom in/out
   - Let PARIMA collect decisions for 5-10 minutes

5. **Export logs**:
   - Click the "Copy Logs" button in the UI
   - Paste into a text editor and save as `parima_decisions_log.csv`

6. **Train the model** with your collected data

## Model Training Options

### Model Types

The training script supports three model types:

- **`random_forest`** (default) - Good for non-linear relationships, handles mixed data types well
- **`svm`** - Good for complex decision boundaries, can be slower
- **`logistic`** - Simple linear model, fast and interpretable

### Command-Line Arguments

```bash
python3 train_model.py [OPTIONS]
```

**Options:**

- `--data PATH` - Path to CSV file with training data
- `--synthetic` - Generate synthetic training data instead
- `--samples N` - Number of synthetic samples (default: 1000)
- `--model-type TYPE` - Model type: `random_forest`, `svm`, or `logistic` (default: `random_forest`)
- `--output PATH` - Output path for model file (default: `../ml_models/PARIMA/model_checkpoint.pkl`)

### Example Commands:

```bash
# Train with synthetic data (1000 samples)
python3 train_model.py --synthetic --samples 1000

# Train with synthetic data (5000 samples)
python3 train_model.py --synthetic --samples 5000

# Train with real logged data
python3 train_model.py --data ../parima_decisions_log.csv

# Train with SVM instead of RandomForest
python3 train_model.py --synthetic --model-type svm

# Train and save to custom location
python3 train_model.py --synthetic --output ../ml_models/PARIMA/custom_model.pkl

# Train logistic model with real data
python3 train_model.py --data logs.csv --model-type logistic
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

## After Training

### 1. Verify Model File

Check that the model file was created:

```bash
ls -lh ml_models/PARIMA/model_checkpoint.pkl
```

### 2. Restart Backend

The backend automatically loads the model on startup:

```bash
PORT=5001 python3 backend/parima_api.py
```

### 3. Verify Model Loaded

Check the health endpoint:

```bash
curl http://localhost:5001/health
```

Should return:
```json
{"model_loaded": true, "model_path": "...", "status": "healthy"}
```

### 4. Test in Frontend

- Refresh your browser
- Check console for: `"PARIMA model loaded successfully"`
- Load a VTP file and watch PARIMA decisions
- Compare fallback (always LOD 1) vs. model predictions

## Understanding Model Output

The training script prints:

- **Training/Test Split**: How data was divided
- **Feature Dimensions**: Should be 35
- **Model Accuracy**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score per LOD class
- **Feature Importance**: Top 5 most important features (RandomForest only)

### Example Output:

```
Training samples: 800
Test samples: 200
Feature dimensions: 35

Training random_forest model...

Model Accuracy: 0.8750

Classification Report:
              precision    recall  f1-score   support

        LOD 0       0.90      0.85      0.87        60
        LOD 1       0.82      0.88      0.85        80
        LOD 2       0.91      0.83      0.87        60

    accuracy                           0.88       200
   macro avg       0.88      0.85      0.86       200
weighted avg       0.88      0.88      0.88       200

Top 5 Most Important Features:
  deviceFPS: 0.2845
  meanVisibleDistance: 0.1923
  frustumCoverage: 0.1567
  deviceCPULoad: 0.0989
  traj_0_vel: 0.0543

Model saved to: ../ml_models/PARIMA/model_checkpoint.pkl
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
- Check CSV has valid LOD values (0, 1, or 2)
- Ensure no NaN or infinite values in feature columns
- Collect more data if needed

### Model Accuracy is Low

**Possible Causes**:
- Not enough training data (collect more)
- Synthetic data doesn't match real patterns (use real data)
- Feature ranges don't match real usage (check feature distributions)

**Solutions**:
- Collect more real training data
- Try different model types
- Adjust model hyperparameters

### Backend Still Shows "model_loaded: false"

**Check**:
1. Model file exists at correct path
2. File permissions are correct
3. Model file is valid pickle format
4. Check backend logs for error messages

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
- Balance data across LOD classes (try to have similar counts for LOD 0, 1, 2)

### Model Selection
- Start with `random_forest` - good default
- Try `logistic` for faster inference if needed
- Use `svm` for complex patterns (but slower training)

### Iterative Improvement
1. Train initial model with synthetic data
2. Deploy and collect real data
3. Retrain with real data
4. Compare performance
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

Add cross-validation for better evaluation:

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

