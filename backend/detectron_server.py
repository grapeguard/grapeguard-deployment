# Simplified server for testing Render deployment
# simple_detectron_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock disease detector for testing
class MockGrapeDiseaseDetector:
    def __init__(self):
        self.class_names = [
            "Karpa (Anthracnose)",
            "Bhuri (Powdery mildew)", 
            "Bokadlela (Borer Infestation)",
            "Davnya (Downey Mildew)",
            "Healthy"
        ]
        self.detectron2_loaded = True  # Mock as loaded
        
    def predict(self, image_data, include_visualization=True):
        # Mock prediction - always return healthy for now
        return {
            "predictions": [{
                "class_id": 5,
                "class_name": "Healthy",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200]
            }],
            "visualization": None,
            "image_info": {"width": 800, "height": 600},
            "processing_time": 0.5,
            "model_info": {
                "architecture": "MOCK MODEL FOR TESTING",
                "status": "testing_deployment"
            }
        }

detector = MockGrapeDiseaseDetector()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "detectron2_available": False,
        "detectron2_loaded": True,
        "model_exists": True,
        "classes": detector.class_names,
        "environment": "render_testing",
        "note": "Mock server for testing deployment"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Mock prediction
        result = detector.predict(data['image'])
        result['timestamp'] = time.time()
        result['server_status'] = "render_deployment_test"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "type": "mock_prediction_error"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "GrapeGuard Backend API",
        "status": "running",
        "endpoints": ["/health", "/predict"],
        "version": "render_test_v1"
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Mock GrapeGuard Backend for Render Testing")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)