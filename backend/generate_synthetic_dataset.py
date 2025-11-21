#!/usr/bin/env python3
"""
Generate balanced synthetic training dataset and save to CSV
This creates a more balanced dataset with all LOD levels (0-5)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from train_model import generate_synthetic_data
import argparse
from datetime import datetime

def generate_balanced_synthetic_data(n_samples_per_lod=200, seed=42):
    """
    Generate balanced synthetic data with equal samples per LOD level
    
    Args:
        n_samples_per_lod: Number of samples per LOD level (0-5)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns matching the training log format
    """
    np.random.seed(seed)
    
    total_samples = n_samples_per_lod * 6  # 6 LOD levels
    print(f"Generating {total_samples} synthetic samples ({n_samples_per_lod} per LOD level)...")
    
    all_data = []
    
    # Generate data for each LOD level with appropriate feature ranges
    lod_configs = {
        0: {  # Highest detail - Excellent conditions
            'fps_range': (55, 120),
            'cpu_range': (0.0, 0.3),
            'distance_range': (10, 150),
            'coverage_range': (0.8, 1.0),
            'occlusion_range': (0.0, 0.2)
        },
        1: {  # High detail - Very good conditions
            'fps_range': (50, 80),
            'cpu_range': (0.2, 0.4),
            'distance_range': (50, 200),
            'coverage_range': (0.7, 0.95),
            'occlusion_range': (0.0, 0.3)
        },
        2: {  # Medium-high detail - Good conditions
            'fps_range': (40, 70),
            'cpu_range': (0.3, 0.5),
            'distance_range': (100, 300),
            'coverage_range': (0.6, 0.9),
            'occlusion_range': (0.0, 0.4)
        },
        3: {  # Medium-low detail - Moderate conditions
            'fps_range': (30, 55),
            'cpu_range': (0.4, 0.6),
            'distance_range': (200, 400),
            'coverage_range': (0.5, 0.8),
            'occlusion_range': (0.1, 0.5)
        },
        4: {  # Low detail - Poor conditions
            'fps_range': (25, 45),
            'cpu_range': (0.5, 0.7),
            'distance_range': (300, 600),
            'coverage_range': (0.4, 0.7),
            'occlusion_range': (0.2, 0.6)
        },
        5: {  # Lowest detail - Very poor conditions
            'fps_range': (20, 35),
            'cpu_range': (0.6, 1.0),
            'distance_range': (400, 1000),
            'coverage_range': (0.3, 0.6),
            'occlusion_range': (0.3, 0.7)
        }
    }
    
    for lod in range(6):
        config = lod_configs[lod]
        
        for i in range(n_samples_per_lod):
            # Generate features based on LOD-specific ranges
            frustumCoverage = np.random.uniform(*config['coverage_range'])
            occlusionRatio = np.random.uniform(*config['occlusion_range'])
            meanVisibleDistance = np.random.uniform(*config['distance_range'])
            deviceFPS = np.random.uniform(*config['fps_range'])
            deviceCPULoad = np.random.uniform(*config['cpu_range'])
            deviceGPULoad = np.random.uniform(0.0, 1.0)
            
            # Generate trajectory features (10 points * 3 features)
            viewportVelocity = np.random.uniform(-50, 50)
            viewportAcceleration = np.random.uniform(-100, 100)
            viewportAngularVelocity = np.random.uniform(-5, 5)
            
            # Create timestamp (simulate sequential data)
            timestamp = int(datetime.now().timestamp() * 1000) + (lod * n_samples_per_lod + i) * 3000
            
            # Create row matching CSV format
            row = {
                'timestamp': timestamp,
                'frustumCoverage': frustumCoverage,
                'occlusionRatio': occlusionRatio,
                'meanVisibleDistance': meanVisibleDistance,
                'deviceFPS': deviceFPS,
                'deviceCPULoad': deviceCPULoad,
                'deviceGPULoad': deviceGPULoad,
                'viewportVelocity': viewportVelocity,
                'viewportAcceleration': viewportAcceleration,
                'viewportAngularVelocity': viewportAngularVelocity,
                'decisionLOD': lod,
                'decisionTiles': '',
                'latencyMs': np.random.uniform(10, 100),
                'fpsAfterDecision': deviceFPS + np.random.uniform(-5, 5),
                'memoryMB': np.random.uniform(50, 500),
                'jsHeapMemoryMB': np.random.uniform(50, 300),
                'memoryPeakMB': np.random.uniform(100, 600),
                'jsHeapPeakMB': np.random.uniform(100, 400),
                'fpsImprovement': np.random.uniform(-10, 10),
                'qualityScore': 1.0 - (lod / 5.0) * 0.4,  # Higher LOD = better quality
                'fpsPerGPUPercent': deviceFPS / max(deviceGPULoad, 0.01),
                'fpsPerMemoryMB': deviceFPS / max(np.random.uniform(50, 500), 1),
                'inAcceptableRange': 1,
                'lodChangeFromPrevious': 0
            }
            
            all_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle to mix LOD levels
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Sort by timestamp to maintain temporal order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate balanced synthetic training dataset')
    parser.add_argument('--samples-per-lod', type=int, default=200,
                       help='Number of samples per LOD level (default: 200)')
    parser.add_argument('--output', type=str, 
                       default='../data/training_logs/synthetic_training_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--merge-with', type=str, default=None,
                       help='Optional: Path to existing CSV to merge with')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Generating Balanced Synthetic Training Dataset")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    df_synthetic = generate_balanced_synthetic_data(
        n_samples_per_lod=args.samples_per_lod,
        seed=args.seed
    )
    
    print(f"✅ Generated {len(df_synthetic)} synthetic samples")
    print()
    
    # Show distribution
    print("LOD Distribution:")
    lod_counts = df_synthetic['decisionLOD'].value_counts().sort_index()
    for lod, count in lod_counts.items():
        pct = (count / len(df_synthetic)) * 100
        print(f"  LOD {lod}: {count:4d} samples ({pct:5.1f}%)")
    print()
    
    # Merge with existing data if specified
    if args.merge_with and os.path.exists(args.merge_with):
        print(f"📊 Merging with existing data: {args.merge_with}")
        df_existing = pd.read_csv(args.merge_with)
        print(f"   Existing samples: {len(df_existing)}")
        
        # Combine and remove duplicates based on timestamp
        df_combined = pd.concat([df_existing, df_synthetic], ignore_index=True)
        initial_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='first')
        final_count = len(df_combined)
        
        print(f"   Combined: {initial_count} samples")
        print(f"   After deduplication: {final_count} samples")
        print()
        
        # Sort by timestamp
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
        df_synthetic = df_combined
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)
    
    print(f"✅ Saved to: {output_path}")
    print(f"   Total samples: {len(df_synthetic)}")
    print()
    
    # Show final distribution if merged
    if args.merge_with and os.path.exists(args.merge_with):
        print("Final LOD Distribution (after merge):")
        lod_counts = df_synthetic['decisionLOD'].value_counts().sort_index()
        for lod, count in lod_counts.items():
            pct = (count / len(df_synthetic)) * 100
            print(f"  LOD {lod}: {count:4d} samples ({pct:5.1f}%)")
        print()
    
    print("=" * 80)
    print("✅ Synthetic dataset generation complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()

