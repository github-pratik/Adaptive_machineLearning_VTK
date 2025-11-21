#!/usr/bin/env python3
"""
Diagnostic script to analyze why model always predicts LOD 2
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def analyze_model():
    """Analyze the trained model and training data"""
    
    # Load model
    model_path = Path('../model_comparison/random_forest_model.pkl')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("=" * 80)
    print("Model Diagnostic Analysis")
    print("=" * 80)
    print()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Classes: {model.classes_}")
    print(f"   Number of classes: {len(model.classes_)}")
    print()
    
    # Load training data
    data_path = Path('../data/training_logs/parima_decisions_log.csv')
    if not data_path.exists():
        print(f"❌ Training data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Training data loaded: {len(df)} samples")
    print()
    
    # Analyze LOD distribution
    print("LOD Distribution in Training Data:")
    lod_counts = df['decisionLOD'].value_counts().sort_index()
    for lod, count in lod_counts.items():
        pct = (count / len(df)) * 100
        print(f"  LOD {lod}: {count:4d} samples ({pct:5.1f}%)")
    print()
    
    # Check feature ranges
    print("Feature Ranges in Training Data:")
    features = ['frustumCoverage', 'occlusionRatio', 'meanVisibleDistance', 
                'deviceFPS', 'deviceCPULoad']
    for feat in features:
        if feat in df.columns:
            print(f"  {feat}:")
            print(f"    Min: {df[feat].min():.3f}, Max: {df[feat].max():.3f}, "
                  f"Mean: {df[feat].mean():.3f}, Std: {df[feat].std():.3f}")
    print()
    
    # Test model on training data
    from train_model import load_training_data
    X, y = load_training_data(str(data_path))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    print("Model Performance on Test Set:")
    print(classification_report(y_test, y_pred, target_names=[f'LOD {c}' for c in model.classes_]))
    print()
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print()
    
    # Analyze predictions by feature ranges
    print("Prediction Analysis by Feature Ranges:")
    print("-" * 80)
    
    # Group by FPS ranges
    fps_col_idx = 3  # deviceFPS is 4th feature (index 3)
    fps_ranges = [
        (0, 300, "Low (0-300)"),
        (300, 600, "Medium (300-600)"),
        (600, 1000, "High (600-1000)"),
        (1000, float('inf'), "Very High (1000+)")
    ]
    
    for min_fps, max_fps, label in fps_ranges:
        mask = (X_test[:, fps_col_idx] >= min_fps) & (X_test[:, fps_col_idx] < max_fps)
        if mask.sum() > 0:
            preds_in_range = y_pred[mask]
            unique, counts = np.unique(preds_in_range, return_counts=True)
            print(f"\nFPS {label}: {mask.sum()} samples")
            for lod, count in zip(unique, counts):
                pct = (count / len(preds_in_range)) * 100
                print(f"  LOD {lod}: {count} ({pct:.1f}%)")
    
    # Distance ranges
    dist_col_idx = 2  # meanVisibleDistance is 3rd feature (index 2)
    dist_ranges = [
        (0, 100, "Close (0-100)"),
        (100, 500, "Medium (100-500)"),
        (500, 1000, "Far (500-1000)"),
        (1000, float('inf'), "Very Far (1000+)")
    ]
    
    print("\n" + "-" * 80)
    for min_dist, max_dist, label in dist_ranges:
        mask = (X_test[:, dist_col_idx] >= min_dist) & (X_test[:, dist_col_idx] < max_dist)
        if mask.sum() > 0:
            preds_in_range = y_pred[mask]
            unique, counts = np.unique(preds_in_range, return_counts=True)
            print(f"\nDistance {label}: {mask.sum()} samples")
            for lod, count in zip(unique, counts):
                pct = (count / len(preds_in_range)) * 100
                print(f"  LOD {lod}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("1. Collect more diverse training data with different scenarios:")
    print("   - Low FPS scenarios (force LOD 4-5)")
    print("   - High FPS scenarios (allow LOD 0-2)")
    print("   - Different distances (close/far)")
    print("   - Different camera movements (fast/slow)")
    print()
    print("2. Retrain model with balanced data across all LOD levels")
    print()
    print("3. Check if features are varying enough during runtime")
    print("   - Monitor FPS, distance, and other key features")
    print("   - Ensure features are in ranges seen during training")

if __name__ == '__main__':
    analyze_model()

