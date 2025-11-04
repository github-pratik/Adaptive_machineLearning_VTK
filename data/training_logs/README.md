# Training Data Logs

This directory contains CSV files with collected PARIMA decision logs for training models.

## Directory Structure

```
data/
└── training_logs/
    ├── parima_decisions_log.csv       # Collected decision logs
    ├── parima_decisions_log_2024.csv  # Dated logs
    └── README.md                      # This file
```

## Usage

1. **Save CSV files here**: When you export logs from the browser, save them to this directory
2. **Train models**: Use these CSV files to train models:
   ```bash
   cd backend
   python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
   ```

## File Naming Convention

- `parima_decisions_log.csv` - Latest/default log file
- `parima_decisions_log_YYYY-MM-DD.csv` - Dated logs for different sessions
- `parima_decisions_log_sessionN.csv` - Multiple training sessions

## Example

```bash
# After collecting data in browser:
# 1. Export logs (click "Copy Logs" button)
# 2. Save to: data/training_logs/my_training_data.csv
# 3. Train model:
python3 backend/train_model.py --data data/training_logs/my_training_data.csv
```

