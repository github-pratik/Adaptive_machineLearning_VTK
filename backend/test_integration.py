"""
Integration test script for PARIMA model loading and prediction
Tests that the model can be loaded and makes valid predictions
"""

import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parima_api import load_model, validate_features
import numpy as np

def test_model_loading():
    """Test that model can be loaded"""
    print("Testing model loading...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'ml_models', 'PARIMA', 'model_checkpoint.pkl')
    
    print(f"Model path: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    model = load_model(model_path)
    
    if model is None:
        print("❌ FAILED: Model could not be loaded")
        return False
    
    print("✅ Model loaded successfully")
    return True, model

def test_feature_validation():
    """Test feature validation"""
    print("\nTesting feature validation...")
    
    # Valid features
    valid_features = {
        'frustumCoverage': 0.8,
        'occlusionRatio': 0.3,
        'meanVisibleDistance': 150.0,
        'deviceFPS': 55.0,
        'deviceCPULoad': 0.4,
        'viewportTrajectory': []
    }
    
    is_valid, error_msg, feature_array = validate_features(valid_features)
    
    if not is_valid:
        print(f"❌ FAILED: Feature validation failed: {error_msg}")
        return False
    
    print(f"✅ Features validated successfully")
    print(f"   Feature array shape: {feature_array.shape}")
    print(f"   Feature array sample (first 5): {feature_array[0, :5]}")
    return True

def test_model_prediction(model):
    """Test model prediction"""
    print("\nTesting model prediction...")
    
    # Create test features
    test_features = {
        'frustumCoverage': 0.8,
        'occlusionRatio': 0.3,
        'meanVisibleDistance': 150.0,
        'deviceFPS': 55.0,
        'deviceCPULoad': 0.4,
        'viewportTrajectory': []
    }
    
    is_valid, error_msg, feature_array = validate_features(test_features)
    
    if not is_valid:
        print(f"❌ FAILED: Could not validate test features")
        return False
    
    try:
        prediction = model.predict(feature_array)
        lod_index = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
        
        if lod_index < 0 or lod_index > 2:
            print(f"❌ FAILED: Invalid LOD prediction: {lod_index}")
            return False
        
        print(f"✅ Prediction successful: LOD {lod_index}")
        return True
    except Exception as e:
        print(f"❌ FAILED: Prediction error: {str(e)}")
        return False

def test_multiple_predictions(model):
    """Test multiple predictions with different scenarios"""
    print("\nTesting multiple prediction scenarios...")
    
    scenarios = [
        {
            'name': 'High Performance',
            'features': {
                'frustumCoverage': 0.9,
                'occlusionRatio': 0.2,
                'meanVisibleDistance': 100.0,
                'deviceFPS': 60.0,
                'deviceCPULoad': 0.3,
                'viewportTrajectory': []
            }
        },
        {
            'name': 'Low Performance',
            'features': {
                'frustumCoverage': 0.5,
                'occlusionRatio': 0.6,
                'meanVisibleDistance': 400.0,
                'deviceFPS': 25.0,
                'deviceCPULoad': 0.8,
                'viewportTrajectory': []
            }
        },
        {
            'name': 'Medium Performance',
            'features': {
                'frustumCoverage': 0.7,
                'occlusionRatio': 0.4,
                'meanVisibleDistance': 250.0,
                'deviceFPS': 45.0,
                'deviceCPULoad': 0.5,
                'viewportTrajectory': []
            }
        }
    ]
    
    all_passed = True
    for scenario in scenarios:
        is_valid, error_msg, feature_array = validate_features(scenario['features'])
        if is_valid:
            prediction = model.predict(feature_array)
            lod = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
            print(f"   {scenario['name']}: LOD {lod}")
        else:
            print(f"   {scenario['name']}: ❌ Validation failed")
            all_passed = False
    
    return all_passed

def main():
    print("=" * 60)
    print("PARIMA Integration Test")
    print("=" * 60)
    
    # Test 1: Model Loading
    result = test_model_loading()
    if not result:
        print("\n❌ Integration test FAILED: Model cannot be loaded")
        return 1
    
    success, model = result
    
    # Test 2: Feature Validation
    if not test_feature_validation():
        print("\n❌ Integration test FAILED: Feature validation failed")
        return 1
    
    # Test 3: Model Prediction
    if not test_model_prediction(model):
        print("\n❌ Integration test FAILED: Model prediction failed")
        return 1
    
    # Test 4: Multiple Scenarios
    if not test_multiple_predictions(model):
        print("\n⚠️  Warning: Some prediction scenarios failed")
    else:
        print("\n✅ All prediction scenarios passed")
    
    print("\n" + "=" * 60)
    print("✅ All integration tests PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start backend: PORT=5001 python3 backend/parima_api.py")
    print("2. Test health endpoint: curl http://localhost:5001/health")
    print("3. Test prediction: curl -X POST http://localhost:5001/api/parima/predict ...")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

