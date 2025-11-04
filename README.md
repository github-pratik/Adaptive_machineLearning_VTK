# 🌐 CIA_Web - Collaborative Immersive Analysis on the Web

**A web-based, real-time collaborative platform for immersive scientific data visualization and analysis with intelligent adaptive streaming.**

CIA_Web leverages **VTK.js**, **WebXR**, **TensorFlow.js**, and **PARIMA** (Predictive Adaptive Rendering for Immersive Media Applications) to support multi-user interaction, high-dimensional data exploration, and intelligent Level-of-Detail (LOD) management in both desktop and VR environments.

---

## ✨ Features

- 🎨 **3D Visualization**: High-performance rendering of VTP (VTK PolyData) files
- 🥽 **WebXR/VR Support**: Immersive experiences in virtual reality
- 📊 **Dimensionality Reduction**: PCA, t-SNE, and UMAP with TensorFlow.js
- 🤖 **PARIMA Adaptive Streaming**: ML-powered intelligent LOD selection
- 📈 **Real-Time Metrics Dashboard**: Live FPS, GPU, and Distance monitoring
- 🔄 **Real-Time Collaboration**: Multi-user editing with Yjs
- 📝 **Decision Logging**: CSV export for analysis and model training
- 🎯 **Adaptive Performance**: Automatically adjusts quality based on device capabilities

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16+ recommended)
- **npm** or **yarn**
- **Python 3.7+** (for PARIMA backend)
- **Modern browser** with WebGL support (Chrome, Firefox, Edge)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Machine_CIA_Web-main
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies (for PARIMA):**
   ```bash
   cd backend
   pip3 install -r requirements.txt
   cd ..
   ```

4. **Start the development server:**
   ```bash
   npm run start
   ```
   This automatically opens the app in your browser at `http://localhost:8080`

5. **Upload a VTP file** from the `vtp_files/` folder to begin visualizing

---

## 📂 Project Structure

```
Machine_CIA_Web-main/
├── src/                      # Frontend source code
│   ├── index.js             # Main application entry point
│   ├── feature_extractor.js # PARIMA feature extraction
│   ├── tile_manager.js      # Tile-based streaming manager
│   ├── logger.js            # Decision logging system
│   ├── controller.html      # Control panel UI
│   └── index.html           # Main HTML template
│
├── ml_adapter/              # ML model adapters
│   └── parima_adapter.js    # PARIMA backend API client
│
├── ml_models/               # Machine learning models
│   └── PARIMA/
│       ├── model_checkpoint.pkl  # Trained PARIMA model
│       └── ...                  # Reference implementations
│
├── backend/                  # Python backend API
│   ├── parima_api.py        # Flask API server
│   ├── train_model.py       # Model training script
│   ├── test_integration.py  # Integration tests
│   ├── test_api.py          # API endpoint tests
│   ├── verify_setup.sh      # Setup verification
│   ├── start_backend.sh     # Startup helper
│   ├── requirements.txt     # Python dependencies
│   ├── README_TRAINING.md   # Training guide
│   └── TESTING.md           # Testing guide
│
├── tiles/                    # Pre-tiled VTP files (user-provided)
│   └── {model_name}/
│       ├── lod0/            # Highest detail
│       ├── lod1/
│       ├── lod2/
│       ├── lod3/
│       ├── lod4/
│       └── lod5/            # Lowest detail
│
├── vtp_files/               # Sample VTP datasets
├── data/                     # Training data and logs
│   └── training_logs/       # Collected decision logs
│
├── config.json              # PARIMA configuration
├── package.json             # Node.js dependencies
├── webpack.config.js        # Webpack configuration
├── README.md                 # This file
└── CURRENT_IMP.md           # Implementation details
```

---

## 🥽 WebXR/VR Support

### Supported Browsers

- **Google Chrome** (Recommended) - WebXR enabled by default
- **Microsoft Edge** - WebXR enabled by default
- **Firefox Nightly** - Requires enabling `dom.vr.enabled` in `about:config`

### VR Setup

1. **With VR Headset:**
   - Connect your VR headset (Oculus, HTC Vive, etc.)
   - Open the application in a supported browser
   - Click **"Send To VR"** button
   - Experience immersive 3D visualization

2. **Without VR Headset (Testing):**
   - Install [Immersive Web Emulator Extension](https://chromewebstore.google.com/detail/immersive-web-emulator/cgffilbpcibhmcfbgggfhfolhkfbhmik)
   - Open Chrome DevTools → WebXR tab
   - Simulate VR interactions

---

## 🤖 PARIMA Adaptive Streaming

PARIMA (Predictive Adaptive Rendering for Immersive Media Applications) uses machine learning to automatically select optimal Level-of-Detail (LOD) levels for 3D geometry based on real-time performance metrics, visibility, and user behavior.

### Features

- **6 LOD Levels** (0-5): From highest detail (LOD 0) to lowest detail (LOD 5)
- **Intelligent Prediction**: ML model predicts optimal LOD every 500ms
- **Adaptive Performance**: Maintains target FPS by adjusting quality
- **Real-Time Monitoring**: Live metrics dashboard in control panel

### Setup Guide

#### 1. Python Backend Setup

1. **Install dependencies:**
   ```bash
   cd backend
   pip3 install -r requirements.txt
   ```

2. **Train or place model:**
   ```bash
   # Option 1: Train with synthetic data
   python3 train_model.py --synthetic --samples 1000
   
   # Option 2: Train with real data
   python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
   ```
   
   The trained model will be saved to `ml_models/PARIMA/model_checkpoint.pkl`

3. **Start backend API:**
   ```bash
   # Using npm script
   npm run parima:backend
   
   # Or manually
   PORT=5001 python3 backend/parima_api.py
   ```
   
   The server runs on `http://localhost:5001` (or 5000) by default.

#### 2. Configuration

Edit `config.json` to configure PARIMA:

```json
{
  "parima": {
    "enabled": true,
    "apiUrl": "http://localhost:5001/api/parima/predict",
    "modelPath": "./ml_models/PARIMA/model_checkpoint.pkl",
    "featureSampleIntervalMs": 5000,
    "viewportHistorySize": 10,
    "tiles": {
      "basePath": "./tiles",
      "lodLevels": ["lod0", "lod1", "lod2", "lod3", "lod4", "lod5"]
    },
    "logging": {
      "enabled": false,
      "logFile": "parima_decisions_log.csv"
    }
  }
}
```

**Configuration Options:**
- `enabled`: Enable/disable PARIMA streaming
- `apiUrl`: Backend API endpoint URL
- `featureSampleIntervalMs`: Decision frequency in milliseconds (default: 5000ms)
- `viewportHistorySize`: Number of camera states to track (default: 10)
- `tiles.basePath`: Base directory for tile files
- `tiles.lodLevels`: Available LOD level names (6 levels: 0-5)
- `logging.enabled`: Enable automatic CSV downloads
- `logging.logFile`: Output filename for CSV logs

#### 3. Usage

1. **Start Backend** (if not already running):
   ```bash
   npm run parima:backend
   ```

2. **Start Frontend:**
   ```bash
   npm run start
   ```

3. **Monitor Real-Time Metrics:**
   - Check the left control panel for live metrics:
     - **FPS**: Frames per second (color-coded: green ≥50, orange 30-50, red <30)
     - **Distance**: Camera distance to model (blue)
     - **GPU Load**: Estimated GPU load percentage (color-coded)

4. **View PARIMA Decisions:**
   - Check browser console for decision logs
   - Each decision shows: `PARIMA decision made: LOD X (latency: Yms)`

#### 4. Collecting Training Data

To collect real-world data for model training:

1. **Enable logging in `config.json`:**
   ```json
   "logging": {
     "enabled": true,
     "logFile": "parima_decisions_log.csv"
   }
   ```

2. **Use the application** for 10-15 minutes:
   - Load different VTP files
   - Move camera around
   - Interact with the 3D model

3. **Export logs:**
   - When buffer reaches 100 entries, CSV downloads automatically
   - Or click "Copy Logs" button in the UI
   - Save CSV files to `data/training_logs/`

4. **Train model with real data:**
   ```bash
   cd backend
   python3 train_model.py --data ../data/training_logs/parima_decisions_log.csv
   ```

### PARIMA Features Collected

**Visibility Metrics:**
- **Frustum Coverage** (0-1): Percentage of geometry visible in camera view
- **Occlusion Ratio** (0-1): Estimate of occluded vs visible geometry
- **Mean Visible Distance**: Average distance from camera to visible points

**Device Metrics:**
- **Device FPS**: Current rendering frames per second
- **Device CPU Load** (0-1): Estimated CPU load based on memory and FPS

**Viewport Trajectory:**
- **Velocity**: Linear camera movement speed
- **Acceleration**: Rate of change in velocity
- **Angular Velocity**: Rotation speed of viewport
- **History**: Last 10 camera states for trajectory prediction

### Decision Logging

When logging is enabled, PARIMA logs each decision to CSV with:
- Timestamp
- All feature values (5 base + 30 trajectory features)
- Decision (LOD level 0-5)
- Latency (API response time)
- Performance outcomes (FPS, memory usage after decision)

**CSV Format:**
```csv
timestamp,frustumCoverage,occlusionRatio,meanVisibleDistance,deviceFPS,deviceCPULoad,viewportVelocity,viewportAcceleration,viewportAngularVelocity,decisionLOD,decisionTiles,latencyMs,fpsAfterDecision,memoryMB
```

### Testing PARIMA

**Quick Verification:**
```bash
# Verify all components
npm run parima:verify

# Test model loading
npm run parima:test

# Test API endpoints
npm run parima:test-api
```

**Full Integration Test:**
1. Start backend: `npm run parima:backend`
2. Start frontend: `npm run start`
3. Open browser console (F12)
4. Check `parimaEnabled` is `true`
5. Load a VTP file
6. Watch for PARIMA decision logs

For detailed testing instructions, see `backend/TESTING.md`.

---

## 📊 Real-Time Metrics Dashboard

The application includes a **real-time metrics dashboard** integrated into the left control panel, displaying:

- **FPS** (Frames Per Second): Current rendering performance
  - 🟢 Green: ≥ 50 FPS (excellent)
  - 🟠 Orange: 30-50 FPS (good)
  - 🔴 Red: < 30 FPS (needs optimization)

- **Distance**: Camera distance to model (in world units)
  - 🔵 Blue display

- **GPU Load**: Estimated GPU utilization percentage
  - 🟢 Green: < 50% (low load)
  - 🟠 Orange: 50-75% (moderate load)
  - 🔴 Red: > 75% (high load)

**Updates every 500ms** with live values from the visualization.

---

## 📈 Dimensionality Reduction

CIA_Web supports three dimensionality reduction techniques:

1. **PCA (Principal Component Analysis)** - TensorFlow.js implementation
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** - Pure JavaScript
3. **UMAP (Uniform Manifold Approximation and Projection)** - Pure JavaScript

**Usage:**
1. Load a VTP file
2. Select reduction method from control panel
3. Choose target dimensions (2D or 3D)
4. Click "Toggle Reduction" to apply
5. Visualize reduced data in new space

---

## 🛠️ Development

### Available Scripts

```bash
# Start development server
npm run start

# Build for production
npm run build

# PARIMA Backend
npm run parima:backend      # Start backend API server
npm run parima:test         # Test model integration
npm run parima:test-api     # Test API endpoints
npm run parima:verify       # Verify setup
```

### Adding New Features

- **Frontend**: Edit files in `src/`
- **Backend**: Edit files in `backend/`
- **Configuration**: Update `config.json`

### Debugging

**Frontend:**
- Open browser DevTools (F12)
- Check console for logs
- Inspect `window.parimaAdapter`, `window.parimaConfig`, etc.

**Backend:**
- Check Python console output
- Test API with: `curl http://localhost:5001/health`
- View API logs for errors

---

## 🐛 Troubleshooting

### PARIMA Not Working

1. **Check backend is running:**
   ```bash
   curl http://localhost:5001/health
   ```
   Should return: `{"model_loaded": true, "status": "healthy"}`

2. **Verify model file exists:**
   ```bash
   ls -l ml_models/PARIMA/model_checkpoint.pkl
   ```

3. **Check browser console** for error messages

4. **Verify configuration:**
   - `config.json` has `parima.enabled: true`
   - `apiUrl` matches backend port

### Tiles Not Loading

- **Tile files not implemented yet**: Currently in development
- System works without tiles using fallback LOD decisions
- To implement tiles:
  1. Create tile directory structure
  2. Generate LOD versions of your models
  3. Place tiles in `tiles/{model_name}/lod{N}/`

### Port Conflicts

If port 5000 is in use (e.g., macOS AirPlay Receiver):
```bash
# Use different port
PORT=5001 python3 backend/parima_api.py

# Update config.json accordingly
"apiUrl": "http://localhost:5001/api/parima/predict"
```

### Model Training Issues

1. **Check Python version:** `python3 --version` (should be 3.7+)
2. **Install dependencies:** `pip3 install -r backend/requirements.txt`
3. **Check data format:** CSV should have correct column headers
4. **See training guide:** `backend/README_TRAINING.md`

---

## 📚 Documentation

- **Main README**: This file
- **Implementation Details**: `CURRENT_IMP.md`
- **Training Guide**: `backend/README_TRAINING.md`
- **Testing Guide**: `backend/TESTING.md`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

- **VTK.js** for 3D visualization
- **TensorFlow.js** for machine learning
- **PARIMA** research and implementation
- **WebXR** for VR support

---

## 🔮 Future Enhancements

- [ ] Frustum-based tile selection
- [ ] Automated tile generation tools
- [ ] Advanced feature engineering
- [ ] Model versioning system
- [ ] Performance analytics dashboard
- [ ] Auto-retraining pipeline
- [ ] Cloud-based model serving

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review `CURRENT_IMP.md` for implementation details

---

**Last Updated**: 2024
**Version**: 1.0.0
