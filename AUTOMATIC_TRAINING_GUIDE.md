# Automatic Training Guide

## Overview

The PARIMA backend now supports **automatic model training**! The system can automatically retrain the model based on collected data, and all training events are logged for tracking.

---

## Features

### ✅ Automatic Training
- **Background Worker**: Automatically checks for new training data at configured intervals
- **Smart Training**: Only trains when enough samples are available
- **Non-Blocking**: Training happens in background, doesn't interrupt predictions

### ✅ Training Logging
- **Event Logging**: All training events are logged to a history file
- **Status Tracking**: Track when models were updated, accuracy, and sample counts
- **API Endpoints**: Check training status and history via REST API

### ✅ Manual Training
- **API Endpoint**: Trigger training manually via HTTP POST
- **Force Training**: Option to force training even with fewer samples
- **Custom Paths**: Override CSV path and model type

---

## Configuration

### config.json Settings

Add the following to your `config.json`:

```json
{
  "parima": {
    "training": {
      "autoTrain": true,                    // Enable automatic training
      "autoTrainIntervalHours": 24,         // Check every 24 hours
      "minSamplesForTraining": 100,         // Minimum samples required
      "logFile": "./data/training_logs/parima_decisions_log.csv",
      "modelType": "random_forest",         // Model type to train
      "trainingLogFile": "./data/training_logs/training_history.log"
    }
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `autoTrain` | `false` | Enable/disable automatic training |
| `autoTrainIntervalHours` | `24` | Hours between automatic training checks |
| `minSamplesForTraining` | `100` | Minimum number of samples required before training |
| `logFile` | `./data/training_logs/parima_decisions_log.csv` | Path to CSV training data |
| `modelType` | `random_forest` | Model type: `random_forest`, `svm`, or `logistic` |
| `trainingLogFile` | `./data/training_logs/training_history.log` | Path to training history log |

---

## How It Works

### Automatic Training Flow

```
┌─────────────────────────────────────────────────┐
│  Backend Starts                                  │
│  - Loads model                                   │
│  - Reads config.json                             │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│  Auto-Training Thread Started                    │
│  (if autoTrain: true)                           │
└──────────────────┬──────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│  Every N Hours (autoTrainIntervalHours)      │
│                                               │
│  1. Check CSV file exists                    │
│  2. Count samples in CSV                     │
│  3. Compare with minSamplesForTraining      │
└──────────────────┬───────────────────────────┘
                   ↓
         ┌─────────┴─────────┐
         │                   │
    Enough Samples?      Not Enough
         │                   │
         ↓                   ↓
┌─────────────────┐  ┌─────────────────┐
│ Start Training  │  │ Skip Training   │
│ - Load data     │  │ (Log skipped)   │
│ - Train model   │  │                  │
│ - Save model    │  │                  │
│ - Reload model  │  │                  │
│ - Log event     │  │                  │
└────────┬────────┘  └─────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│  Model Updated!                                  │
│  - Log to training_history.log                  │
│  - Continue serving predictions                 │
└─────────────────────────────────────────────────┘
```

---

## API Endpoints

### 1. Manual Training Endpoint

**POST** `/api/parima/train`

Trigger training manually.

**Request Body (Optional):**
```json
{
  "force": false,                    // Force training even if < min samples
  "csv_path": "/path/to/custom.csv", // Override CSV path
  "model_type": "svm"                // Override model type
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training started in background",
  "status": "training_in_progress"
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/api/parima/train \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

### 2. Training Status Endpoint

**GET** `/api/parima/training/status`

Get current training status and history.

**Response:**
```json
{
  "training_in_progress": false,
  "last_training_time": "2024-01-15T10:30:00",
  "auto_training_enabled": true,
  "auto_training_interval_hours": 24,
  "min_samples_required": 100,
  "recent_training_history": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "event_type": "TRAINING_COMPLETED",
      "message": "Model training completed successfully",
      "details": {
        "accuracy": 0.85,
        "n_samples": 500,
        "n_test_samples": 100
      }
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:5001/api/parima/training/status
```

### 3. Health Check (Updated)

**GET** `/health`

Now includes training information:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/path/to/model.pkl",
  "last_training_time": "2024-01-15T10:30:00",
  "training_in_progress": false,
  "auto_training_enabled": true
}
```

---

## Training Log File

All training events are logged to `training_history.log` (configurable).

### Log Format

Each line is a JSON object:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "event_type": "TRAINING_COMPLETED",
  "message": "Model training completed successfully",
  "details": {
    "accuracy": 0.85,
    "n_samples": 500,
    "n_test_samples": 100,
    "model_path": "/path/to/model.pkl"
  }
}
```

### Event Types

- `TRAINING_STARTED`: Training process began
- `TRAINING_COMPLETED`: Training finished successfully
- `TRAINING_FAILED`: Training encountered an error
- `TRAINING_SKIPPED`: Training skipped (not enough samples, file not found, etc.)

### Example Log Entries

```json
{"timestamp": "2024-01-15T10:30:00", "event_type": "TRAINING_STARTED", "message": "Starting model training with 500 samples", "details": {"csv_path": "./data/training_logs/parima_decisions_log.csv", "model_type": "random_forest", "n_samples": 500}}
{"timestamp": "2024-01-15T10:30:15", "event_type": "TRAINING_COMPLETED", "message": "Model training completed successfully", "details": {"accuracy": 0.85, "n_samples": 400, "n_test_samples": 100, "model_path": "./ml_models/PARIMA/model_checkpoint.pkl"}}
{"timestamp": "2024-01-15T11:00:00", "event_type": "TRAINING_SKIPPED", "message": "Not enough samples: 50 < 100", "details": {"n_samples": 50, "min_required": 100}}
```

---

## Usage Examples

### Enable Automatic Training

1. **Update config.json:**
   ```json
   {
     "parima": {
       "training": {
         "autoTrain": true,
         "autoTrainIntervalHours": 24,
         "minSamplesForTraining": 100
       }
     }
   }
   ```

2. **Restart backend:**
   ```bash
   PORT=5001 python3 backend/parima_api.py
   ```

3. **Check status:**
   ```bash
   curl http://localhost:5001/api/parima/training/status
   ```

### Manual Training

**Trigger training immediately:**
```bash
curl -X POST http://localhost:5001/api/parima/train
```

**Force training with fewer samples:**
```bash
curl -X POST http://localhost:5001/api/parima/train \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

**Use custom CSV file:**
```bash
curl -X POST http://localhost:5001/api/parima/train \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "./data/training_logs/custom_data.csv"}'
```

### View Training History

**Read log file directly:**
```bash
cat data/training_logs/training_history.log | jq .
```

**Or use API:**
```bash
curl http://localhost:5001/api/parima/training/status | jq .recent_training_history
```

---

## Monitoring

### Check if Training is Running

```bash
curl http://localhost:5001/api/parima/training/status | jq .training_in_progress
```

### View Last Training Time

```bash
curl http://localhost:5001/api/parima/training/status | jq .last_training_time
```

### View Training History

```bash
curl http://localhost:5001/api/parima/training/status | jq .recent_training_history
```

---

## Troubleshooting

### Training Not Running Automatically

1. **Check config:**
   ```bash
   cat config.json | jq .parima.training.autoTrain
   ```
   Should be `true`

2. **Check backend logs:**
   Look for "Automatic training thread started" message

3. **Check CSV file exists:**
   ```bash
   ls -lh data/training_logs/parima_decisions_log.csv
   ```

### Training Skipped (Not Enough Samples)

- **Check sample count:**
  ```bash
  wc -l data/training_logs/parima_decisions_log.csv
  ```
  (Subtract 1 for header)

- **Lower minimum:**
  Update `minSamplesForTraining` in config.json

- **Force training:**
  Use manual endpoint with `"force": true`

### Training Failed

1. **Check training log:**
   ```bash
   tail -n 20 data/training_logs/training_history.log
   ```

2. **Check backend logs:**
   Look for error messages

3. **Verify CSV format:**
   Ensure CSV has required columns:
   - `frustumCoverage`
   - `occlusionRatio`
   - `meanVisibleDistance`
   - `deviceFPS`
   - `deviceCPULoad`
   - `decisionLOD`

---

## Best Practices

1. **Start with Manual Training:**
   - Collect data first
   - Train manually to verify everything works
   - Then enable automatic training

2. **Monitor Training Logs:**
   - Regularly check `training_history.log`
   - Watch for accuracy trends
   - Ensure training is happening as expected

3. **Adjust Intervals:**
   - Start with longer intervals (24 hours)
   - Reduce if you collect data quickly
   - Increase if training is too frequent

4. **Set Appropriate Minimums:**
   - Start with 100 samples minimum
   - Increase as you collect more data
   - Higher minimums = better quality models

---

## Summary

✅ **Automatic Training**: Enabled via `autoTrain: true` in config  
✅ **Training Logging**: All events logged to `training_history.log`  
✅ **Manual Training**: Available via `/api/parima/train` endpoint  
✅ **Status Monitoring**: Check status via `/api/parima/training/status`  
✅ **Non-Blocking**: Training happens in background, doesn't interrupt predictions  

The model now automatically improves itself as you collect more data! 🚀

