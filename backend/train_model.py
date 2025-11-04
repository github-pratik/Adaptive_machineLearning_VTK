"""
Train PARIMA LOD Prediction Model
Trains a scikit-learn model to predict LOD levels from feature vectors
"""

import numpy as np
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse

def load_training_data(csv_path):
    """
    Load training data from CSV log file
    
    Args:
        csv_path: Path to CSV file with logged decisions
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (LOD decisions)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Extract features (first 5 base features)
    feature_columns = [
        'frustumCoverage',
        'occlusionRatio',
        'meanVisibleDistance',
        'deviceFPS',
        'deviceCPULoad'
    ]
    
    # Check if required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X_base = df[feature_columns].values
    
    # Extract trajectory features if available
    # Note: In CSV, trajectory might not be stored separately
    # We'll pad with zeros for now, or you can enhance logging to include trajectory
    n_samples = X_base.shape[0]
    n_trajectory_features = 30  # 10 points * 3 features
    
    # Try to extract trajectory from CSV if logged, otherwise use zeros
    trajectory_features = np.zeros((n_samples, n_trajectory_features))
    
    # Check for trajectory columns in CSV
    traj_cols = ['viewportVelocity', 'viewportAcceleration', 'viewportAngularVelocity']
    if all(col in df.columns for col in traj_cols):
        # Use the last logged trajectory values (simplified - you may want to enhance this)
        # For each sample, try to build trajectory from recent history
        for i in range(n_samples):
            # Try to get trajectory features from current and recent rows
            traj_idx = min(i, n_samples - 1)
            if traj_idx >= 0:
                # Fill trajectory backwards from current position
                for j in range(min(10, n_samples - traj_idx)):
                    if traj_idx + j < n_samples:
                        trajectory_features[i, j*3] = df.iloc[traj_idx + j]['viewportVelocity'] if 'viewportVelocity' in df.columns else 0.0
                        trajectory_features[i, j*3+1] = df.iloc[traj_idx + j]['viewportAcceleration'] if 'viewportAcceleration' in df.columns else 0.0
                        trajectory_features[i, j*3+2] = df.iloc[traj_idx + j]['viewportAngularVelocity'] if 'viewportAngularVelocity' in df.columns else 0.0
                # Fill remaining with zeros
                for j in range(min(10, n_samples - traj_idx), 10):
                    trajectory_features[i, j*3:j*3+3] = 0.0
    
    # Combine features
    X = np.hstack([X_base, trajectory_features])
    
    # Extract target (LOD decisions)
    if 'decisionLOD' not in df.columns:
        raise ValueError("Column 'decisionLOD' not found in CSV. Make sure logging is enabled and decisions are logged.")
    
    y = df['decisionLOD'].values
    
    # Filter out invalid LOD values (now supports 0-5)
    valid_mask = (y >= 0) & (y <= 5) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Also check for NaN/Inf in features
    feature_valid = np.isfinite(X).all(axis=1)
    X = X[feature_valid]
    y = y[feature_valid]
    
    if len(X) == 0:
        raise ValueError("No valid training samples found after filtering")
    
    return X, y

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic training data for initial model
    Useful for testing or when you don't have real data yet
    
    Args:
        n_samples: Number of synthetic samples to generate
        
    Returns:
        X: Feature matrix
        y: Target labels (LOD decisions)
    """
    np.random.seed(42)
    
    # Generate realistic feature ranges
    X_base = np.column_stack([
        np.random.uniform(0.1, 1.0, n_samples),      # frustumCoverage
        np.random.uniform(0.0, 1.0, n_samples),      # occlusionRatio
        np.random.uniform(10.0, 500.0, n_samples),   # meanVisibleDistance
        np.random.uniform(20.0, 60.0, n_samples),    # deviceFPS
        np.random.uniform(0.0, 1.0, n_samples),      # deviceCPULoad
    ])
    
    # Generate trajectory features (30 features)
    X_traj = np.random.uniform(-10.0, 10.0, (n_samples, 30))
    
    X = np.hstack([X_base, X_traj])
    
    # Generate labels based on heuristics
    # LOD 0 (highest detail) when: excellent conditions
    # LOD 1 (high detail) when: very good conditions
    # LOD 2 (medium-high detail) when: good conditions
    # LOD 3 (medium-low detail) when: moderate conditions
    # LOD 4 (low detail) when: poor conditions
    # LOD 5 (lowest detail) when: very poor conditions
    
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        coverage = X[i, 0]
        distance = X[i, 2]
        fps = X[i, 3]
        cpu = X[i, 4]
        
        # Heuristic-based labeling for 6 LOD levels
        if fps > 55 and cpu < 0.3 and distance < 150 and coverage > 0.8:
            y[i] = 0  # Highest detail (LOD 0) - Excellent conditions
        elif fps > 50 and cpu < 0.4 and distance < 200 and coverage > 0.7:
            y[i] = 1  # High detail (LOD 1) - Very good conditions
        elif fps > 40 and cpu < 0.5 and distance < 300 and coverage > 0.6:
            y[i] = 2  # Medium-high detail (LOD 2) - Good conditions
        elif fps > 30 and cpu < 0.6 and distance < 400:
            y[i] = 3  # Medium-low detail (LOD 3) - Moderate conditions
        elif fps > 25 and cpu < 0.7:
            y[i] = 4  # Low detail (LOD 4) - Poor conditions
        else:
            y[i] = 5  # Lowest detail (LOD 5) - Very poor conditions
    
    return X, y

def train_model(X, y, model_type='random_forest', output_path='../ml_models/PARIMA/model_checkpoint.pkl'):
    """
    Train a model to predict LOD levels
    
    Args:
        X: Feature matrix
        y: Target labels
        model_type: Type of model ('random_forest', 'svm', 'logistic')
        output_path: Where to save the trained model
        
    Returns:
        Trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X.shape[1]}")
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    elif model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type} model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Get unique classes present in test data (support up to 6 LOD levels)
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [f'LOD {cls}' for cls in unique_classes]
    print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes))
    
    # Also show distribution of predictions
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("\nPrediction Distribution:")
    for lod, count in zip(unique_pred, counts_pred):
        print(f"  LOD {lod}: {count} predictions ({100*count/len(y_pred):.1f}%)")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        print("\nTop 5 Most Important Features:")
        feature_names = [
            'frustumCoverage', 'occlusionRatio', 'meanVisibleDistance',
            'deviceFPS', 'deviceCPULoad'
        ] + [f'traj_{i//3}_{["vel","acc","ang"][i%3]}' for i in range(30)]
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {output_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train PARIMA LOD prediction model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV file with training data (from logging)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic training data (for testing)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of synthetic samples (if using --synthetic)')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'svm', 'logistic'],
                        help='Type of model to train')
    parser.add_argument('--output', type=str,
                        default='../ml_models/PARIMA/model_checkpoint.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.synthetic:
        print("Generating synthetic training data...")
        X, y = generate_synthetic_data(args.samples)
        print(f"Generated {len(X)} synthetic samples")
    elif args.data:
        print(f"Loading training data from {args.data}...")
        X, y = load_training_data(args.data)
        print(f"Loaded {len(X)} samples from CSV")
    else:
        print("ERROR: Either --data or --synthetic must be specified")
        print("Usage:")
        print("  python train_model.py --synthetic  # Generate synthetic data")
        print("  python train_model.py --data path/to/logs.csv  # Use real logged data")
        return
    
    # Train model
    model = train_model(X, y, model_type=args.model_type, output_path=args.output)
    
    print("\nTraining complete!")
    print(f"\nTo use this model:")
    print(f"1. Make sure the model file is at: {args.output}")
    print(f"2. Restart the backend: PORT=5001 python3 backend/parima_api.py")
    print(f"3. Check backend health: curl http://localhost:5001/health")

if __name__ == '__main__':
    main()

