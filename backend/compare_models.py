"""
Compare PARIMA (RandomForest) vs LSTM Model Performance
Trains both models and generates comparison report
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from train_model import load_training_data, train_model, prepare_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# TensorFlow imports for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def lstm_cross_validate(X, y, sequence_length=10, n_splits=5, verbose=True):
    """
    Perform time series cross-validation for LSTM model
    
    Returns:
        Dictionary with CV scores and statistics
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    # Prepare sequences
    X_sequences, y_sequences = prepare_sequences(X, y, sequence_length)
    
    # Normalize features
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_sequences.shape
    X_flat = X_sequences.reshape(-1, n_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_sequences_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X_sequences_scaled) // 10))
    cv_scores = []
    fold_details = []
    
    if verbose:
        print(f"\nPerforming {tscv.n_splits}-fold Time Series Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sequences_scaled), 1):
        X_train_fold = X_sequences_scaled[train_idx]
        X_test_fold = X_sequences_scaled[test_idx]
        y_train_fold = y_sequences[train_idx]
        y_test_fold = y_sequences[test_idx]
        
        # Check test set distribution
        unique_test, counts_test = np.unique(y_test_fold, return_counts=True)
        test_dist = {lod: count for lod, count in zip(unique_test, counts_test)}
        
        # Convert to categorical
        n_classes = 6
        y_train_cat = to_categorical(y_train_fold, n_classes)
        y_test_cat = to_categorical(y_test_fold, n_classes)
        
        # Build and train model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        
        model.fit(
            X_train_fold, y_train_cat,
            epochs=30,
            batch_size=32,
            validation_data=(X_test_fold, y_test_cat),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test_fold, y_test_cat, verbose=0)
        cv_scores.append(test_accuracy)
        
        fold_details.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': test_accuracy,
            'test_distribution': test_dist
        })
        
        if verbose:
            print(f"  Fold {fold}: Accuracy = {test_accuracy:.4f} (Train: {len(train_idx)}, Test: {len(test_idx)})")
    
    return {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'fold_details': fold_details,
        'n_folds': len(cv_scores)
    }

def analyze_test_set_distribution(y_train, y_test, model_name="Model"):
    """Analyze and return test set distribution statistics"""
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    train_dist = {lod: count for lod, count in zip(unique_train, counts_train)}
    test_dist = {lod: count for lod, count in zip(unique_test, counts_test)}
    
    # Calculate percentages
    train_pct = {lod: 100*count/len(y_train) for lod, count in train_dist.items()}
    test_pct = {lod: 100*count/len(y_test) for lod, count in test_dist.items()}
    
    # Check for class imbalance issues
    warnings = []
    if len(test_dist) < len(train_dist):
        warnings.append(f"⚠️  Test set missing {len(train_dist) - len(test_dist)} class(es) present in training")
    
    # Check if test set is too homogeneous
    if len(test_dist) == 1:
        warnings.append(f"⚠️  Test set contains only one class - accuracy may be misleading")
    elif max(test_pct.values()) > 90:
        warnings.append(f"⚠️  Test set is highly imbalanced ({max(test_pct.values()):.1f}% one class)")
    
    return {
        'train_distribution': train_dist,
        'test_distribution': test_dist,
        'train_percentages': train_pct,
        'test_percentages': test_pct,
        'warnings': warnings
    }

def compare_models(csv_path, output_dir='./model_comparison'):
    """
    Train and compare RandomForest vs LSTM models
    
    Args:
        csv_path: Path to training data CSV
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PARIMA vs LSTM Model Comparison")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading training data...")
    X, y = load_training_data(csv_path)
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
    print()
    
    # Train RandomForest (PARIMA)
    print("=" * 80)
    print("Training RandomForest Model (PARIMA)")
    print("=" * 80)
    rf_path = os.path.join(output_dir, 'random_forest_model.pkl')
    rf_result = train_model(
        X, y, 
        model_type='random_forest',
        output_path=rf_path,
        verbose=True
    )
    
    # Analyze RandomForest test set distribution
    from sklearn.model_selection import train_test_split
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_test_analysis = analyze_test_set_distribution(y_train_rf, y_test_rf, "RandomForest")
    
    print()
    print("=" * 80)
    print("Training LSTM Model")
    print("=" * 80)
    lstm_path = os.path.join(output_dir, 'lstm_model.pkl')
    lstm_cv_result = None
    lstm_test_analysis = None
    
    try:
        lstm_result = train_model(
            X, y,
            model_type='lstm',
            output_path=lstm_path,
            verbose=True
        )
        
        # Perform cross-validation for more reliable accuracy estimate
        if len(X) >= 50:  # Only do CV if we have enough data
            print("\n" + "=" * 80)
            print("LSTM Cross-Validation Analysis")
            print("=" * 80)
            lstm_cv_result = lstm_cross_validate(X, y, verbose=True)
            
            # Analyze test set distribution from single split
            # We need to recreate the split to analyze it
            from train_model import prepare_sequences
            X_sequences, y_sequences = prepare_sequences(X, y, 10)
            split_idx = int(len(X_sequences) * 0.8)
            y_train_seq = y_sequences[:split_idx]
            y_test_seq = y_sequences[split_idx:]
            lstm_test_analysis = analyze_test_set_distribution(y_train_seq, y_test_seq, "LSTM")
        else:
            print(f"\n⚠️  Skipping cross-validation: Need at least 50 samples (have {len(X)})")
            # Still analyze the test set from the single split
            from train_model import prepare_sequences
            X_sequences, y_sequences = prepare_sequences(X, y, 10)
            split_idx = int(len(X_sequences) * 0.8)
            y_train_seq = y_sequences[:split_idx]
            y_test_seq = y_sequences[split_idx:]
            lstm_test_analysis = analyze_test_set_distribution(y_train_seq, y_test_seq, "LSTM")
            
    except Exception as e:
        print(f"❌ LSTM training failed: {str(e)}")
        print("   Make sure TensorFlow is installed: pip install tensorflow")
        lstm_result = None
    
    # Generate comparison report
    print()
    print("=" * 80)
    print("Generating Comparison Report")
    print("=" * 80)
    
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARIMA (RandomForest) vs LSTM Model Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAINING DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"LOD distribution:\n")
        unique, counts = np.unique(y, return_counts=True)
        for lod, count in zip(unique, counts):
            f.write(f"  LOD {lod}: {count} ({100*count/len(y):.1f}%)\n")
        f.write("\n")
        
        f.write("RANDOM FOREST (PARIMA) RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {rf_result['accuracy']:.4f}\n")
        f.write(f"Training samples: {rf_result['n_samples']}\n")
        f.write(f"Test samples: {rf_result['n_test_samples']}\n")
        f.write(f"Model path: {rf_result['model_path']}\n")
        
        # Test set distribution analysis
        if rf_test_analysis:
            f.write(f"\nTest Set Distribution:\n")
            for lod in sorted(rf_test_analysis['test_distribution'].keys()):
                count = rf_test_analysis['test_distribution'][lod]
                pct = rf_test_analysis['test_percentages'][lod]
                f.write(f"  LOD {lod}: {count} ({pct:.1f}%)\n")
            if rf_test_analysis['warnings']:
                f.write("\n⚠️  Warnings:\n")
                for warning in rf_test_analysis['warnings']:
                    f.write(f"  {warning}\n")
        f.write("\n")
        
        if lstm_result:
            f.write("LSTM RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Single Split Accuracy: {lstm_result['accuracy']:.4f}\n")
            f.write(f"Training samples: {lstm_result['n_samples']}\n")
            f.write(f"Test samples: {lstm_result['n_test_samples']}\n")
            f.write(f"Model path: {lstm_result['model_path']}\n")
            
            # Cross-validation results
            if lstm_cv_result:
                f.write(f"\nCross-Validation Results ({lstm_cv_result['n_folds']} folds):\n")
                f.write(f"  Mean Accuracy: {lstm_cv_result['mean_accuracy']:.4f}\n")
                f.write(f"  Std Deviation: {lstm_cv_result['std_accuracy']:.4f}\n")
                f.write(f"  Min Accuracy: {min(lstm_cv_result['cv_scores']):.4f}\n")
                f.write(f"  Max Accuracy: {max(lstm_cv_result['cv_scores']):.4f}\n")
                f.write(f"\n  Fold Details:\n")
                for fold_info in lstm_cv_result['fold_details']:
                    f.write(f"    Fold {fold_info['fold']}: {fold_info['accuracy']:.4f} "
                           f"(Test: {fold_info['test_size']} samples)\n")
            
            # Test set distribution analysis
            if lstm_test_analysis:
                f.write(f"\nTest Set Distribution (Single Split):\n")
                for lod in sorted(lstm_test_analysis['test_distribution'].keys()):
                    count = lstm_test_analysis['test_distribution'][lod]
                    pct = lstm_test_analysis['test_percentages'][lod]
                    f.write(f"  LOD {lod}: {count} ({pct:.1f}%)\n")
                if lstm_test_analysis['warnings']:
                    f.write("\n⚠️  Warnings:\n")
                    for warning in lstm_test_analysis['warnings']:
                        f.write(f"  {warning}\n")
            f.write("\n")
            
            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")
            
            # Compare single split accuracies
            accuracy_diff = lstm_result['accuracy'] - rf_result['accuracy']
            f.write(f"Single Split Accuracy Difference: {accuracy_diff:+.4f} (LSTM - RandomForest)\n")
            if accuracy_diff > 0:
                if rf_result['accuracy'] > 0:
                    improvement = (accuracy_diff / rf_result['accuracy']) * 100
                    f.write(f"LSTM improvement: {improvement:+.2f}%\n")
                else:
                    f.write(f"LSTM improvement: N/A (RandomForest accuracy is 0)\n")
            else:
                if lstm_result['accuracy'] > 0:
                    improvement = (abs(accuracy_diff) / lstm_result['accuracy']) * 100
                    f.write(f"RandomForest advantage: {improvement:+.2f}%\n")
                else:
                    f.write(f"RandomForest advantage: N/A (LSTM accuracy is 0)\n")
            
            # Compare with CV if available
            if lstm_cv_result:
                f.write(f"\nLSTM Cross-Validation Mean: {lstm_cv_result['mean_accuracy']:.4f} "
                       f"(±{lstm_cv_result['std_accuracy']:.4f})\n")
                cv_diff = lstm_cv_result['mean_accuracy'] - rf_result['accuracy']
                f.write(f"CV vs RandomForest: {cv_diff:+.4f}\n")
                if abs(cv_diff) < 0.05:
                    f.write("  → Models perform similarly when using robust evaluation\n")
            
            f.write("\n")
            
            f.write("RECOMMENDATION\n")
            f.write("-" * 80 + "\n")
            
            # Use CV accuracy if available, otherwise single split
            lstm_acc = lstm_cv_result['mean_accuracy'] if lstm_cv_result else lstm_result['accuracy']
            accuracy_diff_final = lstm_acc - rf_result['accuracy']
            
            # Check for dataset size warnings
            if len(X) < 100:
                f.write(f"⚠️  WARNING: Small dataset ({len(X)} samples)\n")
                f.write("   - Results may not be reliable\n")
                f.write("   - Collect more data (500+ samples recommended)\n")
                f.write("   - Single split accuracy can be misleading\n")
                if lstm_cv_result:
                    f.write("   - Cross-validation provides more reliable estimate\n")
                f.write("\n")
            
            if abs(accuracy_diff_final) > 0.05:  # Significant difference (5%)
                if accuracy_diff_final > 0:
                    f.write("✅ LSTM performs better - consider using LSTM for production\n")
                    f.write("   Benefits: Better temporal understanding, sequence modeling\n")
                else:
                    f.write("✅ RandomForest performs better - current PARIMA is optimal\n")
                    f.write("   Benefits: Faster inference, smaller model size, simpler\n")
            elif abs(accuracy_diff_final) > 0.01:  # Small difference
                f.write("⚠️  Models perform similarly - small accuracy difference\n")
                f.write("   - RandomForest: Faster inference, smaller model size, simpler deployment\n")
                f.write("   - LSTM: Better temporal understanding, more complex, research value\n")
                f.write("   - Consider practical factors (latency, model size) over small accuracy gains\n")
            else:
                f.write("⚠️  Models perform nearly identically\n")
                f.write("   - Choose based on practical considerations:\n")
                f.write("   - RandomForest: Faster inference (~1-5ms), smaller model (~1-5MB)\n")
                f.write("   - LSTM: Temporal understanding, larger model (~10-50MB), slower (~10-50ms)\n")
            
            # Additional warnings
            if lstm_test_analysis and lstm_test_analysis['warnings']:
                f.write("\n⚠️  LSTM Test Set Issues:\n")
                for warning in lstm_test_analysis['warnings']:
                    f.write(f"   {warning}\n")
                f.write("   → Single split accuracy may be unreliable\n")
        else:
            f.write("LSTM RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write("❌ LSTM training failed - see error messages above\n")
            f.write("   Common issues:\n")
            f.write("   - TensorFlow not installed: pip install tensorflow\n")
            f.write("   - Insufficient data: Need at least 30 samples for LSTM\n")
            f.write("   - Memory issues: LSTM requires more memory\n")
    
    print(f"✅ Comparison report saved to: {report_path}")
    print()
    print("=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print(f"\nModels saved to: {output_dir}/")
    print(f"Report: {report_path}")
    print("\nTo use a model, update config.json:")
    print('  "modelType": "random_forest"  or  "modelType": "lstm"')
    print(f'  "modelPath": "{rf_path}"  or  "{lstm_path}"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare RandomForest vs LSTM models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with training data')
    parser.add_argument('--output-dir', type=str, default='./model_comparison',
                       help='Output directory for models and report')
    
    args = parser.parse_args()
    compare_models(args.data, args.output_dir)

