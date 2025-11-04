# PARIMA Model Conversion: From 360-Degree Video to 3D Visualization

## Overview

This document describes how we adapted the original **PARIMA** (Predictive Adaptive Rendering for Immersive Media Applications) model, which was designed for **360-degree video streaming** with only **x, y coordinate inputs**, to work with our **3D visualization project** that requires a completely different feature set and prediction task.

---

## Original PARIMA Model

### Purpose
The original PARIMA model was developed for **360-degree video streaming** to predict future viewport positions (where users will look) in spherical/equirectangular video frames.

### Input Features
- **Only 2 features**: `VIEWPORT_x` and `VIEWPORT_y`
  - X and Y coordinates in equirectangular/spherical projection
  - Representing the user's head orientation/viewport position in 360° space

### Output
- **Predicted viewport coordinates**: Future x, y positions
- Used for **bitrate allocation** to tiles in the video frame
- **Continuous prediction**: Predicts exact coordinates for next frames

### Model Architecture
- **Online Learning**: Used `creme` library for incremental/online learning
- **Time Series**: SARIMAX models for x and y coordinates separately
- **Streaming**: Trained on-the-fly as users watch videos
- **Reference**: Paper "PARIMA: Viewport Adaptive 360-Degree Video Streaming" (WWW '21)

### Key Files (Original)
```
ml_models/PARIMA/
├── Prediction/
│   ├── parima.py        # SARIMAX-based prediction
│   ├── bitrate.py       # Bitrate allocation to tiles
│   └── qoe.py          # Quality of Experience calculation
├── PanoSaliency/       # Convert quaternion to x,y coordinates
└── creme/              # Online learning library modifications
```

---

## Our Project Requirements

### Purpose
We needed PARIMA for **3D mesh visualization** to predict optimal **Level-of-Detail (LOD)** levels for adaptive streaming of 3D geometry.

### Challenges
1. **Different input domain**: 3D visualization vs. 360° video
2. **Different prediction task**: LOD classification vs. viewport coordinates
3. **Different features**: Need visibility, device, and trajectory metrics
4. **Different model type**: Classification vs. regression
5. **Different training**: Batch training vs. online learning

---

## Conversion Process

### 1. Feature Space Transformation

#### Original Features (2D)
```python
# Original: Only x, y coordinates
features = {
    'VIEWPORT_x': x_coordinate,  # Spherical/equirectangular x
    'VIEWPORT_y': y_coordinate   # Spherical/equirectangular y
}
```

#### Converted Features (3D - 35 total)
```javascript
// New: Comprehensive 3D visualization metrics
features = {
    // Visibility Metrics (3)
    frustumCoverage: 0.85,        // Percentage of geometry in view
    occlusionRatio: 0.25,         // Occlusion estimate
    meanVisibleDistance: 156.8,    // Camera distance to model
    
    // Device Metrics (2)
    deviceFPS: 45.2,               // Current rendering FPS
    deviceCPULoad: 0.42,           // Estimated CPU load (0-1)
    
    // Viewport Trajectory (30 = 10 history × 3 features)
    viewportTrajectory: [
        {velocity: 2.5, acceleration: 0.1, angularVelocity: 0.8},
        {velocity: 2.3, acceleration: -0.2, angularVelocity: 0.9},
        // ... 8 more history points
    ]
}
```

**Total Features**: 5 base + 30 trajectory = **35 features**

### 2. Prediction Task Transformation

#### Original Task: Regression (Continuous)
```python
# Original: Predict continuous viewport coordinates
output = model.predict(features)
# Returns: (x_pred, y_pred) - exact coordinates
```

#### Converted Task: Classification (Discrete)
```python
# New: Predict discrete LOD level
output = model.predict(features)
# Returns: LOD level (0, 1, 2, 3, 4, or 5)
# Where:
#   LOD 0 = Highest detail (best quality)
#   LOD 1 = High detail
#   LOD 2 = Medium-high detail
#   LOD 3 = Medium-low detail
#   LOD 4 = Low detail
#   LOD 5 = Lowest detail (best performance)
```

### 3. Model Architecture Change

#### Original Architecture
- **Library**: `creme` (online learning)
- **Model Type**: SARIMAX (Statistical time series)
- **Training**: Incremental/streaming
- **Input Shape**: (1, 2) - single sample with 2 features

```python
# Original architecture
from creme import linear_model
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Separate models for x and y
model_x = SARIMAX(series_x, order=(2,0,1))
model_y = SARIMAX(series_y, order=(3,0,0))
```

#### Converted Architecture
- **Library**: `scikit-learn` (batch learning)
- **Model Type**: RandomForestClassifier (default), SVM, or LogisticRegression
- **Training**: Batch training on collected data
- **Input Shape**: (1, 35) - single sample with 35 features

```python
# Converted architecture
from sklearn.ensemble import RandomForestClassifier

# Single model for all features
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

### 4. Feature Engineering

#### Original Feature Engineering
```python
# Original: Simple coordinate transformation
# Convert quaternion head orientation → x, y coordinates
# Apply wrapping for spherical coordinates
# Apply logarithmic differencing for time series
series_log_x = np.log(series_x)
series_log_y = np.log(series_y)
diff_x = np.diff(series_log_x)
diff_y = np.diff(series_log_y)
```

#### Converted Feature Engineering
```javascript
// New: Complex 3D feature extraction

// 1. Visibility Metrics
function computeFrustumCoverage() {
    // Calculate percentage of mesh points in camera frustum
    // Sample points and check bounds
    return visibleCount / totalPoints;
}

function computeOcclusionRatio() {
    // Estimate occlusion based on geometry density
    const density = numCells / numPoints;
    return Math.min(1.0, density / 10.0);
}

function computeMeanVisibleDistance() {
    // Distance from camera to focal point
    const dx = cameraPos[0] - focalPoint[0];
    const dy = cameraPos[1] - focalPoint[1];
    const dz = cameraPos[2] - focalPoint[2];
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// 2. Device Metrics
function getDeviceFPS() {
    // Track frame times, calculate smoothed average
    return averageFPS;
}

function getDeviceCPULoad() {
    // Estimate from memory pressure and FPS
    return Math.min(1.0, usedRatio * 1.5);
}

// 3. Viewport Trajectory
function updateViewportTrajectory() {
    // Track camera position, focal point over time
    // Calculate velocity, acceleration, angular velocity
    // Maintain history of last 10 states
    const velocity = Math.sqrt(dx*dx + dy*dy + dz*dz) / dt;
    const acceleration = (velocity - lastVelocity) / dt;
    const angularVelocity = focalPointDistance / dt;
}
```

### 5. Training Data Generation

#### Original Training
- **Source**: Real 360° video head tracking logs
- **Format**: Time series of (x, y) coordinates
- **Preprocessing**: Quaternion → Equirectangular conversion
- **Training**: Online/incremental as users watch

#### Converted Training
- **Source**: 
  1. **Synthetic Data**: Heuristic-based generation
  2. **Real Data**: Collected from frontend usage logs
  
- **Format**: Feature vectors with labels
  ```csv
  frustumCoverage,occlusionRatio,meanVisibleDistance,deviceFPS,deviceCPULoad,...,decisionLOD
  0.85,0.25,156.8,45.2,0.42,...,2
  ```

- **Labeling Heuristics** (Synthetic):
  ```python
  # LOD 0: Excellent conditions
  if fps > 48 and cpu < 0.4 and distance < 200 and coverage > 0.7:
      lod = 0
  # LOD 1: Very good conditions
  elif fps > 42 and cpu < 0.5 and distance < 280 and coverage > 0.65:
      lod = 1
  # ... similar for LOD 2-5
  ```

- **Training**: Batch training on collected data

---

## Implementation Details

### Feature Extraction Module (`src/feature_extractor.js`)

**Key Functions**:
1. `computeFrustumCoverage()` - Calculates visible geometry percentage
2. `computeOcclusionRatio()` - Estimates occlusion
3. `computeMeanVisibleDistance()` - Camera distance metric
4. `getDeviceFPS()` - Real-time FPS tracking
5. `getDeviceCPULoad()` - CPU load estimation
6. `updateViewportTrajectory()` - 3D camera movement tracking
7. `collectFeatures()` - Combines all features into 35-feature vector

**Trajectory History**:
- Maintains last 10 camera states
- Calculates velocity, acceleration, angular velocity
- Pads to 30 features (10 × 3) for consistent input size

### Backend API (`backend/parima_api.py`)

**Changes from Original**:
- **Input Validation**: Checks for 35 features (not 2)
- **Feature Preprocessing**: Normalizes and pads trajectory
- **Model Loading**: Loads scikit-learn pickle (not creme)
- **Prediction**: Returns LOD level (not x, y coordinates)
- **Error Handling**: Falls back to LOD 1 if model unavailable

**API Endpoint**:
```python
POST /api/parima/predict
{
    "frustumCoverage": 0.85,
    "occlusionRatio": 0.25,
    "meanVisibleDistance": 156.8,
    "deviceFPS": 45.2,
    "deviceCPULoad": 0.42,
    "viewportTrajectory": [...]
}

Response:
{
    "success": true,
    "decision": {"lod": 2},
    "error": null
}
```

### Model Training (`backend/train_model.py`)

**New Training Pipeline**:
1. **Data Loading**: Parse CSV logs or generate synthetic data
2. **Feature Construction**: Build 35-feature vectors
3. **Label Extraction**: Get LOD labels (0-5)
4. **Train/Test Split**: 80/20 split
5. **Model Training**: Train RandomForest/SVM/LogisticRegression
6. **Evaluation**: Accuracy, classification report
7. **Model Saving**: Save as pickle file

**Usage**:
```bash
# Train with synthetic data
python3 train_model.py --synthetic --samples 1000

# Train with real data
python3 train_model.py --data ../data/training_logs/logs.csv
```

---

## Key Differences Summary

| Aspect | Original PARIMA | Converted PARIMA |
|--------|----------------|------------------|
| **Domain** | 360° Video Streaming | 3D Mesh Visualization |
| **Features** | 2 (x, y coordinates) | 35 (visibility + device + trajectory) |
| **Task** | Regression (continuous) | Classification (discrete) |
| **Output** | Viewport coordinates (x, y) | LOD level (0-5) |
| **Library** | creme (online) | scikit-learn (batch) |
| **Model Type** | SARIMAX (time series) | RandomForest/SVM/Logistic |
| **Training** | Online/incremental | Batch training |
| **Input Format** | Time series (x, y) | Feature vector (35 features) |
| **Prediction Frequency** | Every video frame | Every 5000ms (configurable) |
| **Purpose** | Predict viewport for bitrate | Predict LOD for quality |

---

## Migration Steps Taken

### Step 1: Feature Design
1. Analyzed what metrics matter for 3D visualization
2. Identified visibility, device, and trajectory as key areas
3. Designed 35-feature vector structure
4. Implemented feature extraction functions

### Step 2: Model Architecture Selection
1. Evaluated scikit-learn classifiers
2. Selected RandomForest for initial implementation
3. Implemented training pipeline
4. Created synthetic data generation for testing

### Step 3: Backend Implementation
1. Replaced creme with scikit-learn model loading
2. Updated API to accept new feature format
3. Changed output from coordinates to LOD level
4. Implemented validation and error handling

### Step 4: Frontend Integration
1. Implemented feature extraction from VTK.js scene
2. Created adapter for backend communication
3. Integrated decision loop into rendering pipeline
4. Added logging and metrics dashboard

### Step 5: Testing and Validation
1. Trained initial model with synthetic data
2. Tested end-to-end pipeline
3. Collected real data
4. Retrained with real data

---

## Challenges and Solutions

### Challenge 1: Feature Mismatch
**Problem**: Original model expects 2 features, we need 35.

**Solution**: 
- Completely redesigned feature space
- Built new feature extraction module
- Created new training pipeline from scratch

### Challenge 2: Task Type Change
**Problem**: Regression (continuous) vs. Classification (discrete).

**Solution**:
- Switched from SARIMAX to classification models
- Defined 6 discrete LOD levels (0-5)
- Created heuristics for labeling training data

### Challenge 3: Model Library Change
**Problem**: creme (online) vs. scikit-learn (batch).

**Solution**:
- Switched to batch training approach
- Implemented data collection system
- Created retraining workflow

### Challenge 4: 3D Feature Extraction
**Problem**: Need to extract meaningful features from 3D scene.

**Solution**:
- Used VTK.js APIs for visibility metrics
- Implemented camera trajectory tracking
- Estimated device performance metrics

### Challenge 5: Training Data
**Problem**: No existing training data for our use case.

**Solution**:
- Created synthetic data generator
- Implemented data collection logging
- Built training pipeline for real data

---

## Current Status

### ✅ Completed
- [x] Feature extraction module (35 features)
- [x] Backend API with new feature format
- [x] Model training pipeline
- [x] Frontend integration
- [x] Synthetic data generation
- [x] Real data collection system
- [x] 6 LOD levels support (0-5)
- [x] Metrics dashboard
- [x] End-to-end testing

### 🚧 In Progress / Future
- [ ] Tile file generation tools
- [ ] Frustum-based tile selection
- [ ] Advanced feature engineering
- [ ] Model hyperparameter tuning
- [ ] Real-world performance validation

---

## Usage Example

### Training a Model
```bash
# 1. Generate synthetic training data
cd backend
python3 train_model.py --synthetic --samples 1000 --output ../ml_models/PARIMA/model_checkpoint.pkl

# 2. (Optional) Collect real data
# Enable logging in config.json, use application, collect CSV logs

# 3. Train with real data
python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
```

### Using the Model
```bash
# 1. Start backend
npm run parima:backend

# 2. Start frontend
npm run start

# 3. Use application - model automatically predicts LOD levels
```

---

## Files Modified/Created

### New Files
- `src/feature_extractor.js` - 3D feature extraction
- `backend/train_model.py` - Training pipeline
- `backend/parima_api.py` - Flask API for new model
- `ml_adapter/parima_adapter.js` - Frontend API client
- `src/tile_manager.js` - Tile management (adapted for LOD)
- `src/logger.js` - Decision logging

### Modified Concepts
- Feature space: 2D → 3D (2 features → 35 features)
- Model architecture: SARIMAX → RandomForest
- Training approach: Online → Batch
- Output format: Coordinates → LOD level
- Application domain: Video → 3D Visualization

---

## References

1. **Original PARIMA Paper**: 
   - Chopra, L., et al. "PARIMA: Viewport Adaptive 360-Degree Video Streaming." WWW '21.

2. **Original PARIMA Implementation**:
   - Located in `ml_models/PARIMA/Prediction/`
   - Reference implementation for 360° video streaming

3. **Our Implementation**:
   - Adapted for 3D visualization and LOD prediction
   - Uses scikit-learn for batch training
   - 35-feature input with 6-class LOD output

---

## Conclusion

We successfully converted the PARIMA model from a **360-degree video viewport prediction system** (2 features, continuous output) to a **3D visualization LOD prediction system** (35 features, discrete output). The conversion involved:

1. ✅ Complete redesign of feature space
2. ✅ Change from regression to classification
3. ✅ Switch from online to batch learning
4. ✅ Implementation of 3D-specific feature extraction
5. ✅ Creation of new training pipeline
6. ✅ Full integration into 3D visualization workflow

The adapted system maintains the **adaptive streaming philosophy** of PARIMA while being specifically tailored for **3D mesh visualization** with **intelligent LOD selection**.

---

**Last Updated**: 2024  
**Version**: 1.0.0  
**Authors**: CIA_Web Development Team

