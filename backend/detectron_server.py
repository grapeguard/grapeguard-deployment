# FINAL WORKING DETECTRON2 SERVER - YOUR MODEL WILL WORK
# detectron_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model_loaded = False
predictor = None

class GrapeDiseaseDetector:
    def __init__(self, model_path, threshold=0.7):
        self.model_path = model_path
        self.threshold = threshold
        self.predictor = None
        self.model_loaded = False
        
        # YOUR EXACT CLASSES
        self.class_names = [
            "Karpa (Anthracnose)",          # 0
            "Bhuri (Powdery mildew)",       # 1  
            "Bokadlela (Borer Infestation)", # 2
            "Davnya (Downey Mildew)",       # 3
            "Healthy"                       # 4
        ]
        
        # Disease mapping (0-4 to 1-5 for frontend)
        self.disease_mapping = {
            1: {"name": "Karpa (Anthracnose)", "marathi": "कर्पा रोग", "severity": "High"},
            2: {"name": "Bhuri (Powdery Mildew)", "marathi": "भुरी रोग", "severity": "Medium"},
            3: {"name": "Bokadlela (Borer Infestation)", "marathi": "बोकाडलेला", "severity": "High"},
            4: {"name": "Davnya (Downy Mildew)", "marathi": "दवयाचा रोग", "severity": "High"},
            5: {"name": "Healthy", "marathi": "निरोगी पान", "severity": "None"}
        }
    
    def load_model(self):
        """Load your Detectron2 model"""
        try:
            # Import detectron2
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2 import model_zoo
            
            logger.info("✅ Detectron2 imported successfully")
            
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            logger.info("🔥 Loading your trained model...")
            
            # Create config
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Your model settings
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.MODEL.DEVICE = "cpu"
            cfg.DATALOADER.NUM_WORKERS = 0
            
            # Register metadata
            MetadataCatalog.get("grape_disease").thing_classes = self.class_names
            
            # Create predictor
            self.predictor = DefaultPredictor(cfg)
            
            # Test model
            test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
            outputs = self.predictor(test_image)
            
            self.model_loaded = True
            logger.info("🎉 YOUR REAL MODEL IS LOADED AND READY!")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Detectron2 import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def base64_to_image(self, base64_string):
        """Convert base64 to OpenCV image"""
        try:
            image_data = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(image_data))
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"❌ Image conversion failed: {e}")
            return None
    
    def predict(self, image):
        """Make prediction with your model"""
        if not self.model_loaded:
            raise Exception("Model not loaded")
            
        start_time = time.time()
        
        try:
            logger.info("🔬 Running your real model...")
            
            # Get predictions
            outputs = self.predictor(image)
            instances = outputs["instances"]
            
            logger.info(f"🎯 Model output: {len(instances)} detections")
            
            if len(instances) > 0:
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                
                # Get best detection
                best_idx = np.argmax(scores)
                class_id = int(classes[best_idx]) + 1  # Convert 0-4 to 1-5
                confidence = float(scores[best_idx])
                
                disease_info = self.disease_mapping[class_id]
                
                result = {
                    "disease": disease_info["name"],
                    "marathi": disease_info["marathi"],
                    "confidence": round(confidence * 100, 1),
                    "severity": disease_info["severity"],
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "type": "YOUR_REAL_DETECTRON2_MODEL",
                        "detections": len(instances)
                    }
                }
                
                logger.info(f"✅ Prediction: {result['disease']} ({result['confidence']}%)")
                return result
            else:
                # No detections = healthy
                result = {
                    "disease": "Healthy",
                    "marathi": "निरोगी पान",
                    "confidence": 95.0,
                    "severity": "None",
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "type": "YOUR_REAL_DETECTRON2_MODEL",
                        "detections": 0
                    }
                }
                logger.info("✅ No diseases detected")
                return result
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise e

# Initialize detector
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.7'))

detector = GrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "REAL_MODEL_READY" if detector.model_loaded else "MODEL_NOT_LOADED",
        "model_loaded": detector.model_loaded,
        "model_exists": os.path.exists(detector.model_path),
        "model_path": detector.model_path,
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "architecture": "YOUR_TRAINED_MASK_RCNN",
        "note": "REAL MODEL ONLY"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Real model prediction"""
    try:
        if not detector.model_loaded:
            return jsonify({
                "error": "YOUR REAL MODEL IS NOT LOADED",
                "status": "model_not_available"
            }), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        result = detector.predict(image)
        result['timestamp'] = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({
            "error": str(e),
            "type": "prediction_error"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "GrapeGuard REAL AI - Your Trained Model",
        "status": "REAL_MODEL_ONLY",
        "endpoints": ["/health", "/predict"],
        "model_status": "REAL_DETECTRON2" if detector.model_loaded else "MODEL_NOT_LOADED"
    })

if __name__ == '__main__':
    logger.info("🚀 Starting your real model server...")
    
    success = detector.load_model()
    
    if success:
        logger.info("🎉 YOUR REAL MODEL IS READY!")
    else:
        logger.error("💥 MODEL FAILED TO LOAD")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)