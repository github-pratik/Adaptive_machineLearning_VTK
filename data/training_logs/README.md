# Training Data Logs

This directory contains CSV files with collected PARIMA decision logs for training models. The project supports both **RandomForest (PARIMA)** and **LSTM** models, so you can collect separate datasets for each.

## Directory Structure

```
data/
└── training_logs/
    ├── parima_decisions_log.csv       # PARIMA/RandomForest decision logs
    ├── lstm_decisions_log.csv         # LSTM decision logs
    ├── parima_decisions_log_*.csv     # Dated/backup logs
    ├── lstm_decisions_log_*.csv       # Dated/backup logs
    ├── training_history.log           # Training event log
    └── README.md                      # This file
```

## Data Collection

### Collecting PARIMA (RandomForest) Data

1. **Update `config.json`:**
   ```json
   {
     "parima": {
       "modelPath": "./model_comparison/random_forest_model.pkl",
       "logging": {
         "enabled": true,
         "logFile": "./data/training_logs/parima_decisions_log.csv"
       }
     }
   }
   ```

2. **Start backend and frontend:**
   ```bash
   # Terminal 1: Backend
   PORT=5001 npm run parima:backend
   
   # Terminal 2: Frontend
   npm run start
   ```

3. **Use the application** for 15-20 minutes:
   - Load VTP files
   - Interact with 3D model (rotate, pan, zoom)
   - Vary movements and camera positions

4. **Export logs:**
   - CSV auto-downloads when buffer reaches 100 entries
   - Or click "Copy Logs" button
   - Save to `data/training_logs/parima_decisions_log.csv`

### Collecting LSTM Data

1. **Update `config.json`:**
   ```json
   {
     "parima": {
       "modelPath": "./model_comparison/lstm_model.pkl",
       "logging": {
         "enabled": true,
         "logFile": "./data/training_logs/lstm_decisions_log.csv"
       }
     }
   }
   ```

2. **Restart backend and frontend**

3. **Use the application** for 15-20 minutes (similar interactions as PARIMA)

4. **Export logs:**
   - Save to `data/training_logs/lstm_decisions_log.csv`

## Merging Multiple CSV Files

If you have multiple CSV files from browser downloads (e.g., `_data_training_logs_parima_decisions_log (1).csv`, etc.):

```bash
cd backend
python3 merge_training_data.py \
  --data-dir ../data/training_logs \
  --output parima_decisions_log.csv
```

This will:
- Find all backup CSV files
- Merge them into one file
- Remove duplicates based on timestamp
- Save to `parima_decisions_log.csv`

## Usage

### Train Models

**Train and compare both models:**
```bash
cd backend
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison
```

**Train individual model:**
```bash
cd backend
python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
```

## File Naming Convention

- `parima_decisions_log.csv` - Latest PARIMA/RandomForest log file
- `lstm_decisions_log.csv` - Latest LSTM log file
- `parima_decisions_log_YYYY-MM-DD.csv` - Dated logs for different sessions
- `lstm_decisions_log_YYYY-MM-DD.csv` - Dated LSTM logs
- `_data_training_logs_*.csv` - Browser download backups (can be merged)

## CSV Format

Each CSV file contains the following columns:

```csv
timestamp,frustumCoverage,occlusionRatio,meanVisibleDistance,deviceFPS,deviceCPULoad,deviceGPULoad,viewportVelocity,viewportAcceleration,viewportAngularVelocity,decisionLOD,decisionTiles,latencyMs,fpsAfterDecision,memoryMB,jsHeapMemoryMB,memoryPeakMB,jsHeapPeakMB,fpsImprovement,qualityScore,fpsPerGPUPercent,fpsPerMemoryMB,inAcceptableRange,lodChangeFromPrevious
```

**Key Columns:**
- `timestamp` - Unix timestamp in milliseconds
- `frustumCoverage` - Visibility coverage (0-1)
- `occlusionRatio` - Occlusion estimate (0-1)
- `meanVisibleDistance` - Camera distance
- `deviceFPS` - Frames per second
- `deviceCPULoad` - CPU load (0-1)
- `decisionLOD` - Predicted LOD level (0-5)
- `latencyMs` - API response time
- `fpsAfterDecision` - FPS after decision applied

## Data Quality Tips

### For Best Model Performance:

1. **Diverse Data:**
   - Different VTP files (various complexity)
   - Various camera movements (slow, fast, rotations)
   - Different performance conditions (let FPS vary naturally)
   - Multiple LOD levels (aim for all 6 levels: 0-5)

2. **Sufficient Samples:**
   - Minimum: 300-500 samples per model
   - Good: 500-1000 samples
   - Excellent: 1000+ samples

3. **Balanced Distribution:**
   - Try to have samples for all LOD levels
   - If only seeing LOD 1-2, try larger files or slower hardware

4. **Similar Conditions:**
   - When comparing PARIMA vs LSTM, use similar:
     - VTP files
     - Interaction patterns
     - Collection duration
     - Device/browser

## Verification

### Check Data Quality

```bash
cd backend
python3 -c "
import pandas as pd
df = pd.read_csv('../data/training_logs/parima_decisions_log.csv')
print(f'Total samples: {len(df)}')
print(f'LOD distribution:')
print(df['decisionLOD'].value_counts().sort_index())
print(f'FPS range: {df[\"deviceFPS\"].min():.1f} - {df[\"deviceFPS\"].max():.1f}')
"
```

### Verify Both Datasets

```bash
cd backend
python3 -c "
import pandas as pd

print('PARIMA Dataset:')
parima_df = pd.read_csv('../data/training_logs/parima_decisions_log.csv')
print(f'  Samples: {len(parima_df)}')
print(f'  LODs: {sorted(parima_df[\"decisionLOD\"].unique())}')

print('\nLSTM Dataset:')
lstm_df = pd.read_csv('../data/training_logs/lstm_decisions_log.csv')
print(f'  Samples: {len(lstm_df)}')
print(f'  LODs: {sorted(lstm_df[\"decisionLOD\"].unique())}')
"
```

## Example Workflow

```bash
# 1. Collect PARIMA data (15-20 min)
#    → Save to parima_decisions_log.csv

# 2. Update config for LSTM
#    → Change modelPath and logFile

# 3. Collect LSTM data (15-20 min)
#    → Save to lstm_decisions_log.csv

# 4. Merge any backup files
cd backend
python3 merge_training_data.py --data-dir ../data/training_logs

# 5. Train and compare both models
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison

# 6. Generate comparison plots
python3 generate_model_comparison_plots.py \
  --comparison-dir ../model_comparison \
  --output-dir ../model_comparison_plots \
  --csv-data ../data/training_logs/parima_decisions_log.csv
```

## Troubleshooting

### CSV Files Not Downloading

- Check browser download settings
- Check `config.json`: `"logging.enabled": true`
- Use "Copy Logs" button as alternative

### Not Enough Samples

- Increase collection time (30-60 minutes)
- Load larger/more complex VTP files
- Vary interactions more

### Missing LOD Levels

- Try different VTP files (various complexity)
- Adjust camera positions (zoom in/out)
- Let system struggle naturally (don't force high FPS)

## Related Documentation

- **Data Collection Guide**: `COLLECT_TRAINING_DATA.md`
- **Model Comparison Guide**: `MODEL_COMPARISON_GUIDE.md`
- **Training Guide**: `backend/README_TRAINING.md`
