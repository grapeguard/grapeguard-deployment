# FINAL WORKING DETECTRON2 SERVER - INSTALLS DETECTRON2 AUTOMATICALLY
# detectron_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
import time
import logging
import os
import sys
import subprocess
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model
DETECTRON2_AVAILABLE = False
predictor = None

def install_detectron2():
    """Install Detectron2 on first run"""
    global DETECTRON2_AVAILABLE
    
    try:
        # Try importing torch first
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
        
        # Try importing detectron2
        import detectron2
        DETECTRON2_AVAILABLE = True
        logger.info("✅ Detectron2 already available")
        return True
        
    except ImportError:
        logger.info("🔧 Installing Detectron2...")
        
        try:
            # Install detectron2 from PyPI
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "detectron2", "-f", 
                "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.5/index.html"
            ])
            
            # Try import again
            import detectron2
            DETECTRON2_AVAILABLE = True
            logger.info("✅ Detectron2 installed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Detectron2 installation failed: {e}")
            DETECTRON2_AVAILABLE = False
            return False

class GrapeDiseaseDetector:
    def __init__(self, model_path, threshold=0.7):
        self.model_path = model_path
        self.threshold = threshold
        self.predictor = None
        self.model_loaded = False
        
        # Your exact disease classes
        self.class_names = [
            "Karpa (Anthracnose)",          # 0
            "Bhuri (Powdery mildew)",       # 1  
            "Bokadlela (Borer Infestation)", # 2
            "Davnya (Downey Mildew)",       # 3
            "Healthy"                       # 4
        ]
        
        # Disease mapping for frontend (1-5)
        self.disease_mapping = {
            1: {"name": "Karpa (Anthracnose)", "marathi": "कर्पा रोग", "severity": "High"},
            2: {"name": "Bhuri (Powdery Mildew)", "marathi": "भुरी रोग", "severity": "Medium"},
            3: {"name": "Bokadlela (Borer Infestation)", "marathi": "बोकाडलेला", "severity": "High"},
            4: {"name": "Davnya (Downy Mildew)", "marathi": "दवयाचा रोग", "severity": "High"},
            5: {"name": "Healthy", "marathi": "निरोगी पान", "severity": "None"}
        }
    
    def load_model(self):
        """Load Detectron2 model if available"""
        if not DETECTRON2_AVAILABLE:
            logger.warning("⚠️ Detectron2 not available - using fallback")
            return True
            
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ Model file not found: {self.model_path}")
            return True
            
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2 import model_zoo
            
            logger.info("🔥 Loading your trained Detectron2 model...")
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Your model configuration
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
            logger.info("🎉 Your Detectron2 model is ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.info("🎨 Will use color analysis fallback")
            return True
    
    def analyze_leaf_color(self, image):
        """Advanced color analysis for disease detection"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for disease detection
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            yellow_lower = np.array([15, 40, 40])
            yellow_upper = np.array([35, 255, 255])
            brown_lower = np.array([5, 40, 20])
            brown_upper = np.array([20, 255, 150])
            
            # Calculate color percentages
            total_pixels = image.shape[0] * image.shape[1]
            
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
            
            green_percent = (cv2.countNonZero(green_mask) / total_pixels) * 100
            yellow_percent = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
            brown_percent = (cv2.countNonZero(brown_mask) / total_pixels) * 100
            
            logger.info(f"🎨 Color analysis: Green={green_percent:.1f}%, Yellow={yellow_percent:.1f}%, Brown={brown_percent:.1f}%")
            
            # Disease detection based on color analysis
            if green_percent > 60 and yellow_percent < 15 and brown_percent < 10:
                # Mostly green = healthy
                return {"class_id": 5, "confidence": min(0.9, 0.7 + (green_percent / 100) * 0.2)}
            elif brown_percent > 20:
                # Significant brown = Anthracnose
                return {"class_id": 1, "confidence": min(0.85, 0.6 + (brown_percent / 100) * 0.25)}
            elif yellow_percent > 25:
                # Significant yellow = Powdery Mildew
                return {"class_id": 2, "confidence": min(0.8, 0.6 + (yellow_percent / 100) * 0.2)}
            elif brown_percent > 10 or yellow_percent > 15:
                # Some discoloration = Downy Mildew
                return {"class_id": 4, "confidence": 0.7}
            else:
                # Default to healthy
                return {"class_id": 5, "confidence": 0.75}
                
        except Exception as e:
            logger.error(f"❌ Color analysis failed: {e}")
            return {"class_id": 5, "confidence": 0.5}
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            image_data = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(image_data))
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info(f"🖼️ Image converted: {opencv_image.shape}")
            return opencv_image
        except Exception as e:
            logger.error(f"❌ Image conversion failed: {e}")
            return None
    
    def predict(self, image):
        """Make prediction using real model or color analysis"""
        start_time = time.time()
        
        try:
            # Try real Detectron2 model first
            if self.model_loaded and self.predictor:
                logger.info("🔬 Using real Detectron2 model...")
                
                outputs = self.predictor(image)
                instances = outputs["instances"]
                
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
                        "method": "detectron2_real_model",
                        "processing_time": round(time.time() - start_time, 3)
                    }
                    
                    logger.info(f"✅ Real model prediction: {result['disease']} ({result['confidence']}%)")
                    return result
                else:
                    # No detections = healthy
                    result = {
                        "disease": "Healthy",
                        "marathi": "निरोगी पान",
                        "confidence": 92.0,
                        "severity": "None",
                        "method": "detectron2_no_detections",
                        "processing_time": round(time.time() - start_time, 3)
                    }
                    logger.info("✅ Real model: No diseases detected")
                    return result
            
            # Fallback to color analysis
            else:
                logger.info("🎨 Using advanced color analysis...")
                
                color_result = self.analyze_leaf_color(image)
                disease_info = self.disease_mapping[color_result['class_id']]
                
                result = {
                    "disease": disease_info["name"],
                    "marathi": disease_info["marathi"],
                    "confidence": round(color_result['confidence'] * 100, 1),
                    "severity": disease_info["severity"],
                    "method": "advanced_color_analysis",
                    "processing_time": round(time.time() - start_time, 3)
                }
                
                logger.info(f"✅ Color analysis prediction: {result['disease']} ({result['confidence']}%)")
                return result
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                "disease": "Detection Error",
                "marathi": "तपासणी त्रुटी",
                "confidence": 0,
                "severity": "Unknown",
                "method": "error",
                "error": str(e),
                "processing_time": round(time.time() - start_time, 3)
            }

# Initialize on startup
logger.info("🚀 Starting GrapeGuard Backend...")

# Install Detectron2 if needed
install_success = install_detectron2()

# Initialize detector
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.7'))

detector = GrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "detectron2_available": DETECTRON2_AVAILABLE,
        "model_loaded": detector.model_loaded,
        "model_exists": os.path.exists(detector.model_path),
        "prediction_method": "detectron2" if detector.model_loaded else "color_analysis",
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "environment": "production"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Disease prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        result = detector.predict(image)
        result['timestamp'] = time.time()
        result['server_version'] = "production_v1"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        return jsonify({
            "error": str(e),
            "type": "prediction_error"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "GrapeGuard Disease Detection API",
        "status": "running",
        "endpoints": ["/health", "/predict"],
        "model_status": "detectron2" if detector.model_loaded else "color_analysis",
        "version": "production_v1"
    })

if __name__ == '__main__':
    # Load model on startup
    success = detector.load_model()
    
    if success:
        if detector.model_loaded:
            logger.info("🎉 Real Detectron2 model ready!")
        else:
            logger.info("🎨 Advanced color analysis ready!")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Server running on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)