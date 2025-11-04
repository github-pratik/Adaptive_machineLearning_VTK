"""
PARIMA Backend API Server
Provides HTTP endpoint for PARIMA model inference
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global model variable
parima_model = None
model_path = None

def load_model(model_path):
    """
    Load PARIMA model from pickle file
    
    Args:
        model_path: Path to .pkl model file
        
    Returns:
        Loaded model object or None if failed
    """
    global parima_model
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        logger.info(f"Loading PARIMA model from {model_path}...")
        with open(model_path, 'rb') as f:
            parima_model = pickle.load(f)
        logger.info("Model loaded successfully")
        return parima_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def validate_features(features):
    """
    Validate feature vector matches expected format
    
    Args:
        features: Dictionary containing feature values
        
    Returns:
        (is_valid, error_message, feature_array)
    """
    required_features = [
        'frustumCoverage',
        'occlusionRatio',
        'meanVisibleDistance',
        'deviceFPS',
        'deviceCPULoad'
    ]
    
    # Check for required features
    missing = [f for f in required_features if f not in features]
    if missing:
        return False, f"Missing required features: {missing}", None
    
    # Extract feature values in order
    feature_array = []
    try:
        for feature_name in required_features:
            value = float(features[feature_name])
            if not np.isfinite(value):
                return False, f"Invalid value for {feature_name}: {value}", None
            feature_array.append(value)
        
        # Add viewport trajectory features if available
        if 'viewportTrajectory' in features:
            trajectory = features['viewportTrajectory']
            if isinstance(trajectory, list) and len(trajectory) > 0:
                # Flatten trajectory (velocity, acceleration, etc.)
                for item in trajectory[-10:]:  # Take last 10 if available
                    if isinstance(item, dict):
                        feature_array.extend([
                            item.get('velocity', 0.0),
                            item.get('acceleration', 0.0),
                            item.get('angularVelocity', 0.0)
                        ])
        
        # Pad to expected length if needed (PARIMA models may expect fixed input size)
        # Adjust based on your actual model's expected input size
        expected_size = len(required_features) + 30  # 10 trajectory points * 3 features
        if len(feature_array) < expected_size:
            feature_array.extend([0.0] * (expected_size - len(feature_array)))
        elif len(feature_array) > expected_size:
            feature_array = feature_array[:expected_size]
            
        feature_array = np.array(feature_array).reshape(1, -1)
        return True, None, feature_array
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid feature format: {str(e)}", None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': parima_model is not None,
        'model_path': model_path
    })

@app.route('/api/parima/predict', methods=['POST'])
def predict():
    """
    PARIMA prediction endpoint
    
    Expected JSON body:
    {
        "frustumCoverage": float,
        "occlusionRatio": float,
        "meanVisibleDistance": float,
        "deviceFPS": float,
        "deviceCPULoad": float,
        "viewportTrajectory": [{"velocity": float, "acceleration": float, "angularVelocity": float}, ...]
    }
    
    Returns:
    {
        "success": bool,
        "decision": {
            "lod": int,
            "tiles": [str] (optional)
        },
        "error": str (if failed)
    }
    """
    global parima_model
    
    if parima_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'decision': {'lod': 1}  # Fallback to medium LOD
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'decision': {'lod': 1}
            }), 400
        
        # Validate features
        is_valid, error_msg, feature_array = validate_features(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'decision': {'lod': 1}
            }), 400
        
        # Perform prediction
        logger.info(f"Making prediction with features shape: {feature_array.shape}")
        prediction = parima_model.predict(feature_array)
        
        # Parse prediction result
        # Assumes model returns LOD index (0, 1, 2, etc.)
        lod_index = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
        
        # Clamp LOD to valid range
        lod_index = max(0, min(lod_index, 5))  # Now supports 6 LOD levels (0, 1, 2, 3, 4, 5)
        
        # If model also returns tile list, extract it
        # For now, return LOD only (tile selection can be done client-side)
        decision = {
            'lod': lod_index
        }
        
        logger.info(f"Prediction result: LOD {lod_index}")
        
        return jsonify({
            'success': True,
            'decision': decision
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}',
            'decision': {'lod': 1}  # Fallback
        }), 500

def initialize_app(model_file_path=None):
    """
    Initialize the Flask app and load model
    
    Args:
        model_file_path: Optional path to model file
    """
    global model_path
    
    # Determine model path
    if model_file_path:
        model_path = model_file_path
    else:
        # Default path relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'ml_models', 'PARIMA', 'model_checkpoint.pkl')
    
    # Load model
    if load_model(model_path):
        logger.info("PARIMA API ready")
    else:
        logger.warning("Model not loaded - API will return fallback decisions")

if __name__ == '__main__':
    import sys
    
    # Check for command-line model path
    model_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    initialize_app(model_path_arg)
    
    # Start Flask server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting PARIMA API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

