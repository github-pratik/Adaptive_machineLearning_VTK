# PARIMA Testing Guide

Complete guide for testing the PARIMA implementation.

## Quick Start Testing

### 1. Verify Setup

Run the verification script:

```bash
bash backend/verify_setup.sh
```

This checks:
- ✅ Model file exists
- ✅ Python dependencies installed
- ✅ Model can be loaded
- ✅ Backend imports work
- ✅ Config file exists
- ✅ Node modules installed

### 2. Test Model Loading

Test that the model can be loaded and makes predictions:

```bash
cd backend
python3 test_integration.py
```

Expected output:
```
✅ Model loaded successfully
✅ Features validated successfully
✅ Prediction successful: LOD X
✅ All prediction scenarios passed
```

### 3. Test Backend API

**Start the backend** (in Terminal 1):
```bash
PORT=5001 python3 backend/parima_api.py
```

**Test API** (in Terminal 2):
```bash
cd backend
python3 test_api.py
```

Or test manually:
```bash
# Health check
curl http://localhost:5001/health

# Prediction test
curl -X POST http://localhost:5001/api/parima/predict \
  -H "Content-Type: application/json" \
  -d '{
    "frustumCoverage": 0.8,
    "occlusionRatio": 0.3,
    "meanVisibleDistance": 150.0,
    "deviceFPS": 55.0,
    "deviceCPULoad": 0.4,
    "viewportTrajectory": []
  }'
```

### 4. Test Frontend Integration

**Start frontend** (in Terminal 2):
```bash
npm run start
```

**In Browser:**
1. Open `http://localhost:8080`
2. Open Developer Console (F12)
3. Check for PARIMA initialization messages
4. Verify variables:
   ```javascript
   parimaEnabled      // Should be true
   parimaAdapter      // Should show object
   parimaAdapter.isModelAvailable()  // Should be true
   ```
5. Load a VTP file
6. Watch for PARIMA decision logs every 500ms
7. Click "Copy Logs" to export CSV

## Detailed Test Scenarios

### Scenario 1: Model Loading Test

**Purpose**: Verify model file can be loaded

```bash
python3 backend/test_integration.py
```

**Expected**: All tests pass, model loads successfully

### Scenario 2: API Health Test

**Purpose**: Verify backend is running and model is loaded

```bash
curl http://localhost:5001/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "..."
}
```

### Scenario 3: API Prediction Test

**Purpose**: Verify predictions work correctly

```bash
python3 backend/test_api.py
```

**Expected**: All test cases return valid LOD values (0, 1, or 2)

### Scenario 4: Feature Validation Test

**Purpose**: Verify feature validation works

**Test Valid Features**:
```bash
curl -X POST http://localhost:5001/api/parima/predict \
  -H "Content-Type: application/json" \
  -d '{
    "frustumCoverage": 0.8,
    "occlusionRatio": 0.3,
    "meanVisibleDistance": 150.0,
    "deviceFPS": 55.0,
    "deviceCPULoad": 0.4,
    "viewportTrajectory": []
  }'
```

**Expected**: `{"success": true, "decision": {"lod": X}}`

**Test Invalid Features** (missing required):
```bash
curl -X POST http://localhost:5001/api/parima/predict \
  -H "Content-Type: application/json" \
  -d '{"frustumCoverage": 0.8}'
```

**Expected**: `{"success": false, "error": "...", "decision": {"lod": 1}}`

### Scenario 5: Frontend Integration Test

**Steps**:
1. Start backend and frontend
2. Load VTP file
3. Check browser console for:
   - `"PARIMA configuration loaded"`
   - `"PARIMA model loaded successfully"`
   - `"PARIMA streaming started"`
4. Move camera around
5. Verify decisions appear in console
6. Export logs and verify CSV format

### Scenario 6: Performance Test

**Measure**:
- Prediction latency (< 50ms target)
- Decision frequency (every 500ms)
- Memory usage
- FPS impact

**Tools**:
- Browser Performance tab
- Console logs show latency
- Memory profiler

### Scenario 7: Error Handling Test

**Test Cases**:
1. Backend not running → Frontend should fallback gracefully
2. Invalid model file → Backend should start with warnings
3. Missing features → API should return error with fallback
4. Network timeout → Frontend should use fallback LOD

## Test Checklist

### Backend Tests
- [ ] Model file exists and is valid
- [ ] Backend starts successfully
- [ ] Health endpoint returns `model_loaded: true`
- [ ] Prediction endpoint works
- [ ] Predictions are valid LOD values (0, 1, 2)
- [ ] Error handling works (invalid input)
- [ ] Fallback decisions work when model unavailable

### Frontend Tests
- [ ] PARIMA config loads
- [ ] PARIMA adapter initializes
- [ ] Model availability check works
- [ ] Feature extraction works
- [ ] Decisions are made every 500ms
- [ ] Logging captures all features
- [ ] CSV export works
- [ ] Copy Logs button works
- [ ] Fallback works when backend down

### Integration Tests
- [ ] End-to-end: Frontend → Backend → Model → Decision
- [ ] Decision changes based on camera movement
- [ ] Decision changes based on device performance
- [ ] Logs can be used for retraining

## Troubleshooting Tests

### Backend Won't Start

```bash
# Check Python version
python3 --version

# Check dependencies
pip3 list | grep -E "flask|numpy|sklearn|pandas"

# Check model file
ls -lh ml_models/PARIMA/model_checkpoint.pkl

# Test model loading directly
python3 backend/test_integration.py
```

### Model Not Loading

```bash
# Check backend logs for error messages
# Look for:
# - "Model file not found"
# - "Failed to load model"
# - Python version mismatch

# Test model loading
python3 -c "import pickle; pickle.load(open('ml_models/PARIMA/model_checkpoint.pkl', 'rb'))"
```

### Frontend Can't Connect

```bash
# Check backend is running
curl http://localhost:5001/health

# Check CORS (should not be an issue with flask-cors)
# Check browser console for network errors
```

### Predictions Always Same LOD

- Check if model is actually loaded: `parimaAdapter.isModelAvailable()`
- Check feature values are varying
- Check backend logs for prediction errors
- Verify model file is correct (retrain if needed)

## Automated Testing

Run all tests:

```bash
# 1. Verify setup
bash backend/verify_setup.sh

# 2. Test model integration
cd backend && python3 test_integration.py

# 3. Test API (requires backend running)
# Start backend first, then:
python3 backend/test_api.py
```

## Performance Benchmarks

Expected performance:
- **Model Loading**: < 1 second
- **Prediction Latency**: < 50ms
- **Decision Frequency**: Every 500ms
- **Feature Extraction**: < 10ms
- **CSV Log Export**: < 100ms for 100 entries

## Next Steps After Testing

1. **Collect Real Data**: Use the system with real VTP files
2. **Export Logs**: Use "Copy Logs" button
3. **Retrain Model**: `python3 backend/train_model.py --data logs.csv`
4. **Compare Performance**: Synthetic vs. real model
5. **Iterate**: Collect more data, retrain, improve

