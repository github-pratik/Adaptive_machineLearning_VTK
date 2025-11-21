#!/usr/bin/env python3
"""
Merge all _data_training_logs_parima_decisions_log*.csv files into one file.
Removes duplicates based on timestamp and saves to parima_decisions_log.csv
"""

import pandas as pd
import os
import glob
from pathlib import Path

def merge_training_data(
    data_dir='../data/training_logs',
    output_file='parima_decisions_log.csv',
    backup_original=True
):
    """
    Merge all backup CSV files into the main training data file.
    
    Args:
        data_dir: Directory containing CSV files
        output_file: Name of output file
        backup_original: Whether to backup original file before merging
    """
    data_path = Path(data_dir)
    output_path = data_path / output_file
    
    print("=" * 80)
    print("Merging Training Data Files")
    print("=" * 80)
    print()
    
    # Find all backup files matching the pattern
    backup_pattern = str(data_path / '_data_training_logs_parima_decisions_log*.csv')
    backup_files = sorted(glob.glob(backup_pattern))
    
    print(f"Found {len(backup_files)} backup files:")
    for f in backup_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Also include the main file if it exists
    all_files = []
    if output_path.exists():
        all_files.append(str(output_path))
        print(f"✅ Main file found: {output_path.name}")
    all_files.extend(backup_files)
    
    if not all_files:
        print("❌ No CSV files found to merge!")
        return
    
    print(f"\n📊 Reading {len(all_files)} files...")
    
    # Read all CSV files
    all_dataframes = []
    file_stats = []
    
    for file_path in all_files:
        try:
            # Check if file has data (more than just header)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:
                    print(f"  ⚠️  Skipping empty file: {os.path.basename(file_path)}")
                    continue
            
            df = pd.read_csv(file_path)
            if len(df) > 0:
                all_dataframes.append(df)
                file_stats.append({
                    'file': os.path.basename(file_path),
                    'rows': len(df)
                })
                print(f"  ✅ {os.path.basename(file_path):50s} - {len(df):4d} rows")
            else:
                print(f"  ⚠️  Skipping empty dataframe: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ❌ Error reading {os.path.basename(file_path)}: {str(e)}")
    
    if not all_dataframes:
        print("\n❌ No valid data found in any files!")
        return
    
    print(f"\n📦 Combining {len(all_dataframes)} dataframes...")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"   Total rows before deduplication: {len(combined_df)}")
    
    # Remove duplicates based on timestamp (most reliable unique identifier)
    print(f"\n🔍 Removing duplicates (based on timestamp)...")
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
    duplicates_removed = initial_count - len(combined_df)
    
    print(f"   Removed {duplicates_removed} duplicate rows")
    print(f"   Unique rows: {len(combined_df)}")
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Backup original file if it exists and backup is enabled
    if output_path.exists() and backup_original:
        backup_path = data_path / f"{output_file}.backup"
        print(f"\n💾 Backing up original file to: {backup_path.name}")
        import shutil
        shutil.copy2(output_path, backup_path)
    
    # Save merged data
    print(f"\n💾 Saving merged data to: {output_path.name}")
    combined_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Merge Summary")
    print("=" * 80)
    print(f"✅ Successfully merged {len(all_dataframes)} files")
    print(f"✅ Total unique samples: {len(combined_df)}")
    print(f"✅ Duplicates removed: {duplicates_removed}")
    print()
    
    # File statistics
    print("File Statistics:")
    for stat in file_stats:
        print(f"  - {stat['file']:50s}: {stat['rows']:4d} rows")
    print()
    
    # Data quality check
    print("Data Quality:")
    print(f"  - LOD levels: {sorted(combined_df['decisionLOD'].unique())}")
    print(f"  - FPS range: {combined_df['deviceFPS'].min():.2f} - {combined_df['deviceFPS'].max():.2f}")
    print(f"  - Distance range: {combined_df['meanVisibleDistance'].min():.2f} - {combined_df['meanVisibleDistance'].max():.2f}")
    print(f"  - Missing values: {combined_df.isnull().sum().sum()}")
    print()
    
    # LOD distribution
    print("LOD Distribution:")
    lod_counts = combined_df['decisionLOD'].value_counts().sort_index()
    for lod, count in lod_counts.items():
        pct = (count / len(combined_df)) * 100
        print(f"  LOD {lod}: {count:4d} samples ({pct:5.1f}%)")
    print()
    
    print(f"✅ Merge complete! Data saved to: {output_path}")
    print("=" * 80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge training data CSV files')
    parser.add_argument('--data-dir', type=str, default='../data/training_logs',
                       help='Directory containing CSV files')
    parser.add_argument('--output', type=str, default='parima_decisions_log.csv',
                       help='Output filename')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backing up original file')
    
    args = parser.parse_args()
    
    merge_training_data(
        data_dir=args.data_dir,
        output_file=args.output,
        backup_original=not args.no_backup
    )

