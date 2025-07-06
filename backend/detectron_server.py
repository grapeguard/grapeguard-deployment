# Hybrid Detectron2 Server - Real ML with Fallback
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
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Try to import detectron2
DETECTRON2_AVAILABLE = False
try:
    # Add detectron2 to path if it exists
    detectron_path = os.path.join(os.path.dirname(__file__), 'detectron')
    if os.path.exists(detectron_path) and detectron_path not in sys.path:
        sys.path.insert(0, detectron_path)
    
    import detectron2
    from detectron2.detectron2.engine import DefaultPredictor
    from detectron2.detectron2.config import get_cfg
    from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
    logger.info("✅ Detectron2 imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Detectron2 not available: {e}")
    DETECTRON2_AVAILABLE = False

class HybridGrapeDiseaseDetector:
    def __init__(self, model_path, threshold=0.7):
        self.model_path = model_path
        self.threshold = threshold
        self.predictor = None
        self.detectron2_loaded = False
        self.use_fallback = False
        
        # Disease information
        self.class_names = [
            "Karpa (Anthracnose)",
            "Bhuri (Powdery mildew)", 
            "Bokadlela (Borer Infestation)",
            "Davnya (Downey Mildew)",
            "Healthy"
        ]
        
        self.disease_mapping = {
            1: {
                "name": "Karpa (Anthracnose)",
                "marathi": "कर्पा रोग",
                "severity": "High",
                "confidence_threshold": 0.7
            },
            2: {
                "name": "Bhuri (Powdery Mildew)",
                "marathi": "भुरी रोग", 
                "severity": "Medium",
                "confidence_threshold": 0.6
            },
            3: {
                "name": "Bokadlela (Borer Infestation)",
                "marathi": "बोकाडलेला",
                "severity": "High",
                "confidence_threshold": 0.7
            },
            4: {
                "name": "Davnya (Downy Mildew)",
                "marathi": "दवयाचा रोग",
                "severity": "High",
                "confidence_threshold": 0.6
            },
            5: {
                "name": "Healthy",
                "marathi": "निरोगी पान",
                "severity": "None",
                "confidence_threshold": 0.8
            }
        }
    
    def load_model(self):
        """Try to load real detectron2 model, fallback to color analysis"""
        if DETECTRON2_AVAILABLE and os.path.exists(self.model_path):
            try:
                logger.info("🔥 Attempting to load real Detectron2 model...")
                
                cfg = get_cfg()
                # Try to load config
                config_paths = [
                    "detectron/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                ]
                
                config_loaded = False
                for config_path in config_paths:
                    if os.path.exists(config_path):
                        cfg.merge_from_file(config_path)
                        config_loaded = True
                        logger.info(f"✅ Loaded config from: {config_path}")
                        break
                
                if not config_loaded:
                    # Manual config
                    logger.info("⚙️ Using manual config setup")
                    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
                    cfg.MODEL.RESNETS.DEPTH = 50
                
                # Model settings
                cfg.MODEL.WEIGHTS = self.model_path
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
                cfg.MODEL.DEVICE = "cpu"
                cfg.DATALOADER.NUM_WORKERS = 0
                
                # Register metadata
                MetadataCatalog.get("leaf_disease_dataset").thing_classes = self.class_names
                
                # Create predictor
                self.predictor = DefaultPredictor(cfg)
                
                # Test prediction
                test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
                outputs = self.predictor(test_image)
                
                self.detectron2_loaded = True
                logger.info("🎉 Real Detectron2 model loaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Detectron2 loading failed: {e}")
                logger.info("🔄 Falling back to color analysis...")
                self.use_fallback = True
                return True
        else:
            logger.info("🔄 Using color-based fallback detection...")
            self.use_fallback = True
            return True
    
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
            logger.error(f"❌ Image conversion failed: {str(e)}")
            return None
    
    def analyze_leaf_color(self, image):
        """Fallback: Analyze leaf health based on color"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for different conditions
            # Green (healthy)
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            
            # Yellow/Brown (disease indicators)
            yellow_lower = np.array([15, 40, 40])
            yellow_upper = np.array([35, 255, 255])
            
            # Brown (severe disease)
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
            
            # Determine disease based on color analysis
            if green_percent > 60 and yellow_percent < 15 and brown_percent < 10:
                # Mostly green = healthy
                return {
                    "class_id": 5,
                    "confidence": min(0.9, 0.6 + (green_percent / 100) * 0.3),
                    "method": "color_analysis",
                    "color_breakdown": {
                        "green": green_percent,
                        "yellow": yellow_percent,
                        "brown": brown_percent
                    }
                }
            elif yellow_percent > 25 or brown_percent > 15:
                # Significant yellowing/browning
                if brown_percent > yellow_percent:
                    # More brown = severe disease (Anthracnose)
                    return {
                        "class_id": 1,
                        "confidence": min(0.85, 0.6 + (brown_percent / 100) * 0.25),
                        "method": "color_analysis",
                        "color_breakdown": {
                            "green": green_percent,
                            "yellow": yellow_percent,
                            "brown": brown_percent
                        }
                    }
                else:
                    # More yellow = powdery mildew
                    return {
                        "class_id": 2,
                        "confidence": min(0.8, 0.55 + (yellow_percent / 100) * 0.25),
                        "method": "color_analysis",
                        "color_breakdown": {
                            "green": green_percent,
                            "yellow": yellow_percent,
                            "brown": brown_percent
                        }
                    }
            else:
                # Uncertain - default to mildly unhealthy
                return {
                    "class_id": 2,
                    "confidence": 0.6,
                    "method": "color_analysis_uncertain",
                    "color_breakdown": {
                        "green": green_percent,
                        "yellow": yellow_percent,
                        "brown": brown_percent
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Color analysis failed: {e}")
            return {
                "class_id": 5,
                "confidence": 0.5,
                "method": "color_analysis_error",
                "error": str(e)
            }
    
    def predict(self, image, include_visualization=False):
        """Make prediction using real model or fallback"""
        start_time = time.time()
        
        try:
            # Try real detectron2 prediction first
            if self.detectron2_loaded and not self.use_fallback:
                logger.info("🔬 Using real Detectron2 model...")
                outputs = self.predictor(image)
                instances = outputs["instances"]
                
                detections = []
                if len(instances) > 0:
                    scores = instances.scores.cpu().numpy()
                    classes = instances.pred_classes.cpu().numpy()
                    boxes = instances.pred_boxes.tensor.cpu().numpy()
                    
                    for i in range(len(scores)):
                        class_id = int(classes[i])
                        confidence = float(scores[i])
                        bbox = boxes[i].tolist()
                        
                        detections.append({
                            "class_id": class_id + 1,  # Convert 0-4 to 1-5
                            "class_name": self.class_names[class_id],
                            "confidence": confidence,
                            "bbox": bbox,
                            "method": "detectron2"
                        })
                
                if detections:
                    best_detection = max(detections, key=lambda x: x['confidence'])
                    disease_info = self.disease_mapping[best_detection['class_id']]
                    
                    result = {
                        "disease": disease_info["name"],
                        "marathi": disease_info["marathi"],
                        "confidence": round(best_detection['confidence'] * 100, 1),
                        "severity": disease_info["severity"],
                        "method": "detectron2_real_model",
                        "detections": detections,
                        "processing_time": time.time() - start_time
                    }
                else:
                    # No detections = healthy
                    result = {
                        "disease": "Healthy",
                        "marathi": "निरोगी पान",
                        "confidence": 92.5,
                        "severity": "None",
                        "method": "detectron2_no_detections",
                        "processing_time": time.time() - start_time
                    }
                    
                logger.info(f"✅ Real model prediction: {result['disease']} ({result['confidence']}%)")
                return result
            
            # Fallback to color analysis
            else:
                logger.info("🎨 Using color-based analysis...")
                color_result = self.analyze_leaf_color(image)
                disease_info = self.disease_mapping[color_result['class_id']]
                
                result = {
                    "disease": disease_info["name"],
                    "marathi": disease_info["marathi"],
                    "confidence": round(color_result['confidence'] * 100, 1),
                    "severity": disease_info["severity"],
                    "method": "color_analysis_fallback",
                    "color_breakdown": color_result.get('color_breakdown', {}),
                    "processing_time": time.time() - start_time,
                    "note": "Using color-based analysis (Detectron2 not available)"
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
                "processing_time": time.time() - start_time
            }

# Initialize detector
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.7'))

detector = HybridGrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "detectron2_available": DETECTRON2_AVAILABLE,
        "detectron2_loaded": detector.detectron2_loaded,
        "using_fallback": detector.use_fallback,
        "model_exists": os.path.exists(detector.model_path),
        "model_path": detector.model_path,
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "prediction_method": "detectron2" if detector.detectron2_loaded and not detector.use_fallback else "color_analysis",
        "environment": "production_hybrid"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Smart prediction endpoint - real ML or fallback"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        include_visualization = data.get('include_visualization', False)
        result = detector.predict(image, include_visualization)
        
        result['timestamp'] = time.time()
        result['server_info'] = {
            "detectron2_available": DETECTRON2_AVAILABLE,
            "using_real_model": detector.detectron2_loaded and not detector.use_fallback,
            "prediction_method": result.get('method', 'unknown')
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {str(e)}")
        return jsonify({
            "error": str(e),
            "type": "prediction_error"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "GrapeGuard Backend API - Hybrid ML",
        "status": "running",
        "endpoints": ["/health", "/predict"],
        "model_status": "detectron2" if detector.detectron2_loaded and not detector.use_fallback else "color_analysis",
        "version": "hybrid_v1"
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Hybrid GrapeGuard Backend...")
    
    success = detector.load_model()
    
    if success:
        if detector.detectron2_loaded and not detector.use_fallback:
            logger.info("🎉 Real Detectron2 model ready!")
        else:
            logger.info("🎨 Color analysis fallback ready!")
    else:
        logger.error("💥 Model loading failed!")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)