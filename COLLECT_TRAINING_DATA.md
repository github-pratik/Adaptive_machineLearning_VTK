# Collect Training Data Automatically

This guide explains how to collect training data automatically using Option 1.

## Data Directory Structure

All training data is stored in the `data/training_logs/` directory:

```
data/
└── training_logs/
    ├── parima_decisions_log.csv       # Main training data file (used by models)
    ├── parima_decisions_log_*.csv     # Backup files (optional)
    └── training_history.log           # Training event log
```

**Important**: 
- The main data file must be named `parima_decisions_log.csv` and located in `data/training_logs/`
- This is the file path configured in `config.json`
- After downloading CSV from browser, move it to `data/training_logs/parima_decisions_log.csv`

## Quick Start

### Step 1: Start the Backend

```bash
cd /Users/shashikant/Desktop/GMU_HW/cs692_project/Machine_CIA_Web-main
PORT=5001 npm run parima:backend
```

Keep this terminal running. You should see:
```
INFO:__main__:PARIMA API ready
INFO:__main__:Starting PARIMA API server on port 5001
```

### Step 2: Start the Frontend

Open a **new terminal** and run:

```bash
cd /Users/shashikant/Desktop/GMU_HW/cs692_project/Machine_CIA_Web-main
npm run start
```

The application will open in your browser (usually `http://localhost:8080`).

### Step 3: Use the Application

1. **Load a VTP file**:
   - Click "Load VTP File"
   - Select a file from `vtp_files/` (e.g., `Skull.vtp`, `Lungs.vtp`)

2. **Interact with the 3D model**:
   - **Rotate**: Click and drag
   - **Pan**: Right-click and drag (or Shift + drag)
   - **Zoom**: Scroll wheel
   - **Move around**: Explore different angles and distances

3. **Vary your interactions**:
   - Slow movements
   - Fast movements
   - Rotations
   - Zoom in/out
   - Different camera positions

### Step 4: Monitor Data Collection

Watch the **log panel** in the top-right corner of the browser. You'll see messages like:

```
✅ PARIMA decision logged: LOD 1, FPS: 42.3, Memory: 0.0MB (Entries: 1)
✅ PARIMA decision logged: LOD 2, FPS: 41.8, Memory: 0.0MB (Entries: 2)
...
✅ CSV buffer full! (100 entries) CSV file should download automatically...
```

### Step 5: Export Data

**Automatic Download:**
- When the buffer reaches 100 entries, a CSV file will automatically download
- Check your browser's download folder (usually `~/Downloads/`)
- The file will be named: `parima_decisions_log.csv`
- **Important**: Move the downloaded file to `data/training_logs/parima_decisions_log.csv`

**Manual Export:**
- Click the "Copy Logs" button in the UI
- Paste the CSV content into a text editor
- Save as `data/training_logs/parima_decisions_log.csv` (relative to project root)

## Data Collection Tips

### Collect Diverse Data

For best model performance, collect data with:

1. **Different VTP files**:
   - Load multiple files (Skull, Lungs, Bones, etc.)
   - Each file has different geometry complexity

2. **Various camera movements**:
   - Slow, smooth movements
   - Fast, jerky movements
   - Continuous rotations
   - Zoom in/out cycles

3. **Different performance conditions**:
   - Let FPS vary naturally
   - Don't force high/low FPS artificially
   - Let the system adapt

4. **Multiple LOD levels**:
   - The model should use different LODs (0-5)
   - If you only see LOD 1-2, try larger files or slower hardware

### Recommended Collection Time

- **Minimum**: 5-10 minutes (100-200 samples)
- **Good**: 15-30 minutes (300-600 samples)
- **Excellent**: 30-60 minutes (600-1200 samples)

**For reliable LSTM comparison**: Collect 500+ samples

### Check Current Data

All collected data is stored in `data/training_logs/` directory:

```bash
# Navigate to project root
cd /Users/shashikant/Desktop/GMU_HW/cs692_project/Machine_CIA_Web-main

# Count current samples
wc -l data/training_logs/parima_decisions_log.csv

# View data statistics
cd backend
python3 -c "
import pandas as pd
df = pd.read_csv('../data/training_logs/parima_decisions_log.csv')
print(f'Total samples: {len(df)}')
print(f'\nLOD distribution:')
print(df['decisionLOD'].value_counts().sort_index())
print(f'\nFPS range: {df[\"deviceFPS\"].min():.1f} - {df[\"deviceFPS\"].max():.1f}')
"
```

## Configuration

Your `config.json` is set up for automatic collection. The log file path is configured as:

```json
{
  "parima": {
    "enabled": true,
    "logging": {
      "enabled": true,
      "logFile": "./data/training_logs/parima_decisions_log.csv"
    },
    "training": {
      "logFile": "./data/training_logs/parima_decisions_log.csv",
      "autoTrain": true,
      "minSamplesForTraining": 100
    }
  }
}
```

**Note**: The path `./data/training_logs/parima_decisions_log.csv` is relative to the project root directory. Make sure your CSV files are saved in the `data/training_logs/` folder.

**Key Settings:**
- `featureSampleIntervalMs: 3000` - Collects data every 3 seconds
- `maxBufferSize: 100` - Downloads CSV when 100 entries collected
- `autoTrain: true` - Automatically retrains when enough data collected

## Troubleshooting

### No data being collected?

1. **Check PARIMA is enabled**:
   - Look for "PARIMA modules initialized" in browser console
   - Check `config.json`: `"enabled": true`

2. **Check logging is enabled**:
   - Look for "PARIMA decision logging enabled" in browser console
   - Check `config.json`: `"logging.enabled": true`

3. **Check backend is running**:
   ```bash
   curl http://localhost:5001/health
   ```
   Should return: `{"status": "healthy"}`

4. **Check browser console**:
   - Open Developer Tools (F12)
   - Look for errors in Console tab

### CSV not downloading?

1. **Check browser download settings**:
   - Some browsers block automatic downloads
   - Check download permissions for the site

2. **Manual export**:
   - Click "Copy Logs" button
   - Paste into a text editor
   - Save manually to `data/training_logs/parima_decisions_log.csv` (relative to project root)

### Not enough samples?

- **Increase collection time**: Run for 30-60 minutes
- **Load larger files**: More complex geometry = more varied data
- **Vary interactions**: More diverse movements = better training data

## After Collecting Data

### 1. Check Data Quality

The data file should be located at `data/training_logs/parima_decisions_log.csv`:

```bash
# From project root
cd /Users/shashikant/Desktop/GMU_HW/cs692_project/Machine_CIA_Web-main

# Check if file exists
ls -lh data/training_logs/parima_decisions_log.csv

# Analyze data quality
cd backend
python3 -c "
import pandas as pd
df = pd.read_csv('../data/training_logs/parima_decisions_log.csv')
print(f'✅ Total samples: {len(df)}')
print(f'✅ LOD levels: {sorted(df[\"decisionLOD\"].unique())}')
print(f'✅ FPS range: {df[\"deviceFPS\"].min():.1f} - {df[\"deviceFPS\"].max():.1f}')
print(f'✅ Missing values: {df.isnull().sum().sum()}')
"
```

### 2. Train Models

Make sure your data file is in `data/training_logs/parima_decisions_log.csv`:

```bash
# From project root
cd /Users/shashikant/Desktop/GMU_HW/cs692_project/Machine_CIA_Web-main/backend

# Train and compare both models
python3 compare_models.py \
  --data ../data/training_logs/parima_decisions_log.csv \
  --output-dir ../model_comparison
```

**Note**: The path `../data/training_logs/parima_decisions_log.csv` is relative to the `backend/` directory.

### 3. Review Results

```bash
cat model_comparison/model_comparison_report.txt
```

## Quick Reference

**Start collection:**
```bash
# Terminal 1: Backend
PORT=5001 npm run parima:backend

# Terminal 2: Frontend
npm run start
```

**Check progress:**
- Watch browser log panel
- Check entry count in log messages
- CSV downloads at 100, 200, 300... entries

**Stop collection:**
- Close browser tab
- Stop frontend: `Ctrl+C` in frontend terminal
- Stop backend: `Ctrl+C` in backend terminal

**Data location:**
- **Browser downloads**: Files download to `~/Downloads/parima_decisions_log.csv`
- **Training data directory**: All training data should be in `data/training_logs/`
- **Main log file**: `data/training_logs/parima_decisions_log.csv` (this is what the model uses)
- **Backup files**: If you have multiple CSV files, save them in `data/training_logs/` with descriptive names

**Important**: After downloading CSV from browser, move it to `data/training_logs/parima_decisions_log.csv` to ensure the training scripts can find it.

