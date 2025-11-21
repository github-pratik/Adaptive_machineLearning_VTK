# Data Collection Guide: PARIMA vs Baseline Comparison

This guide explains how to collect data for comparing PARIMA adaptive streaming with baseline (no adaptation) performance.

---

## Overview

To create meaningful comparison visualizations, you need:
1. **Baseline Data**: Performance metrics WITHOUT PARIMA (fixed quality)
2. **PARIMA Data**: Performance metrics WITH PARIMA (adaptive streaming)

Both datasets should be collected under similar conditions for fair comparison.

---

## Step-by-Step Data Collection

### Part 1: Collect Baseline Data (Without PARIMA)

#### Step 1: Setup Baseline Configuration

Run the helper script to automatically configure for baseline collection:

```bash
cd backend
python3 collect_baseline_data.py setup
```

This will:
- Backup your current `config.json`
- Disable PARIMA
- Enable logging
- Set log file to `baseline_no_parima.csv`

#### Step 2: Start Backend (Optional - for consistency)

```bash
PORT=5001 npm run parima:backend
```

Note: Backend is optional since PARIMA is disabled, but starting it ensures consistent setup.

#### Step 3: Start Frontend and Collect Data

```bash
npm run start
```

**Important:** Use the application in the same way you'll use it with PARIMA:
- Load the same VTP file
- Perform similar interactions (rotate, zoom, pan)
- Use for the same duration (e.g., 5-10 minutes)
- Try to replicate the same usage patterns

#### Step 4: Export Baseline Data

1. When you have enough data (100+ samples recommended):
   - Click "Copy Logs" button in the UI, OR
   - Wait for automatic CSV download (when buffer reaches 100 entries)

2. Save the CSV file:
   ```bash
   # Move downloaded CSV to training logs directory
   mv ~/Downloads/baseline_no_parima.csv data/training_logs/
   ```

#### Step 5: Restore PARIMA Configuration

```bash
cd backend
python3 collect_baseline_data.py restore
```

This restores your original PARIMA-enabled configuration.

---

### Part 2: Collect PARIMA Data (With Adaptive Streaming)

#### Step 1: Ensure PARIMA is Enabled

Check `config.json`:
```json
{
  "parima": {
    "enabled": true,  // Must be true
    "logging": {
      "enabled": true,  // Enable logging
      "logFile": "parima_decisions_log.csv"
    }
  }
}
```

#### Step 2: Start Backend

```bash
PORT=5001 npm run parima:backend
```

#### Step 3: Start Frontend and Collect Data

```bash
npm run start
```

**Important:** Use the application in the SAME way as baseline:
- Load the same VTP file
- Perform similar interactions
- Use for the same duration
- Try to replicate the same usage patterns

#### Step 4: Export PARIMA Data

1. When you have enough data:
   - Click "Copy Logs" button, OR
   - Wait for automatic CSV download

2. Ensure CSV is saved:
   ```bash
   # Verify file exists
   ls -lh data/training_logs/parima_decisions_log.csv
   ```

---

## Step 3: Generate Comparison Visualizations

Once you have both datasets:

```bash
cd backend
python3 generate_comparison_plots.py \
  --parima-csv ../data/training_logs/parima_decisions_log.csv \
  --baseline-csv ../data/training_logs/baseline_no_parima.csv \
  --output-dir ../comparison_plots
```

This will generate:
1. **FPS Comparison** - Line chart and distribution
2. **Latency Comparison** - Box plots and violin plots
3. **LOD Distribution** - Bar charts showing adaptive behavior
4. **Adaptive Behavior** - LOD changes over time
5. **Performance Improvement** - Before/after metrics
6. **Summary Report** - Text report with statistics

All plots are saved as high-resolution PNG files (300 DPI) ready for presentation.

---

## Quick Reference

### Collect Baseline Data
```bash
# 1. Setup
cd backend
python3 collect_baseline_data.py setup

# 2. Start frontend
npm run start

# 3. Use application, collect data, export CSV

# 4. Restore config
python3 collect_baseline_data.py restore
```

### Collect PARIMA Data
```bash
# 1. Ensure PARIMA enabled in config.json
# 2. Start backend
PORT=5001 npm run parima:backend

# 3. Start frontend
npm run start

# 4. Use application, collect data, export CSV
```

### Generate Comparison
```bash
cd backend
python3 generate_comparison_plots.py \
  --parima-csv ../data/training_logs/parima_decisions_log.csv \
  --baseline-csv ../data/training_logs/baseline_no_parima.csv \
  --output-dir ../comparison_plots
```

---

## Tips for Fair Comparison

1. **Same Duration**: Collect data for the same amount of time in both cases
2. **Same Interactions**: Try to perform similar camera movements and interactions
3. **Same VTP File**: Use the same 3D model file for both tests
4. **Same Device**: Run both tests on the same computer/browser
5. **Similar Conditions**: Avoid other heavy applications running during tests
6. **Enough Samples**: Collect at least 100 samples for meaningful statistics

---

## Troubleshooting

### Baseline CSV has no LOD data
- This is expected! Baseline doesn't make LOD decisions
- The comparison script handles this automatically

### CSV files not found
- Check file paths are correct
- Ensure CSV files are in `data/training_logs/` directory

### Plots look empty
- Verify CSV files have data (check with `head` or `wc -l`)
- Ensure required columns exist (FPS, latency, etc.)

### Import errors
- Install required packages:
  ```bash
  pip3 install matplotlib seaborn scipy
  ```

---

## Output Files

After running the comparison script, you'll have:

```
comparison_plots/
├── 1_fps_comparison.png          # FPS over time and distribution
├── 2_latency_comparison.png       # Latency box plots
├── 3_lod_distribution.png         # LOD decision distribution
├── 4_adaptive_behavior.png        # Adaptive LOD changes over time
├── 5_performance_improvement.png  # Before/after improvement metrics
└── comparison_summary.txt         # Text report with statistics
```

All images are 300 DPI, ready for presentations and papers!

---

## Example Workflow

```bash
# Day 1: Collect Baseline
cd backend
python3 collect_baseline_data.py setup
npm run start
# ... use application for 10 minutes ...
# ... export CSV to data/training_logs/baseline_no_parima.csv ...
python3 collect_baseline_data.py restore

# Day 2: Collect PARIMA Data
# ... ensure PARIMA enabled in config.json ...
PORT=5001 npm run parima:backend
npm run start
# ... use application for 10 minutes ...
# ... CSV already in data/training_logs/parima_decisions_log.csv ...

# Day 3: Generate Comparison
cd backend
python3 generate_comparison_plots.py \
  --parima-csv ../data/training_logs/parima_decisions_log.csv \
  --baseline-csv ../data/training_logs/baseline_no_parima.csv \
  --output-dir ../comparison_plots

# View results
open ../comparison_plots/
```

---

**Ready to create presentation-quality comparison visualizations!** 🎨📊

