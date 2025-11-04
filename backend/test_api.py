"""
Test PARIMA API endpoints
Tests the Flask API for model loading and predictions
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:5001"

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Health check failed: Backend not running")
        print("   Start backend: PORT=5001 python3 backend/parima_api.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    test_cases = [
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
    for test_case in test_cases:
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/api/parima/predict",
                json=test_case['features'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    lod = data.get('decision', {}).get('lod', -1)
                    if 0 <= lod <= 2:
                        print(f"✅ {test_case['name']}: LOD {lod} (latency: {latency:.1f}ms)")
                    else:
                        print(f"❌ {test_case['name']}: Invalid LOD {lod}")
                        all_passed = False
                else:
                    error = data.get('error', 'Unknown error')
                    lod = data.get('decision', {}).get('lod', -1)
                    print(f"⚠️  {test_case['name']}: {error}, fallback LOD {lod}")
            else:
                print(f"❌ {test_case['name']}: HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_case['name']}: {str(e)}")
            all_passed = False
    
    return all_passed

def test_invalid_features():
    """Test error handling with invalid features"""
    print("\nTesting error handling...")
    
    invalid_features = {
        'frustumCoverage': 0.8,
        # Missing required features
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/parima/predict",
            json=invalid_features,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 400:
            data = response.json()
            if not data.get('success'):
                print("✅ Error handling works (returned 400 for invalid input)")
                return True
        
        print("⚠️  Error handling may not be working as expected")
        return False
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("PARIMA API Integration Test")
    print("=" * 60)
    print(f"\nTesting API at: {API_BASE}")
    print("Make sure backend is running: PORT=5001 python3 backend/parima_api.py\n")
    
    # Test 1: Health endpoint
    model_loaded = test_health_endpoint()
    
    if not model_loaded:
        print("\n⚠️  Model not loaded - predictions will use fallback LOD")
        print("   Check backend logs for model loading errors")
    
    # Test 2: Prediction endpoint
    if not test_prediction_endpoint():
        print("\n❌ Some prediction tests failed")
        return 1
    
    # Test 3: Error handling
    test_invalid_features()
    
    print("\n" + "=" * 60)
    print("✅ API tests completed")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

