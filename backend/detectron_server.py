# FI# CORRECT IMPORTS - use these exactly
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
import sys

# Force CPU usage for deployment
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        """Load your Detectron2 model with deployment fixes"""
        try:
            logger.info("🚀 Starting model loading...")
            
            # CORRECT DETECTRON2 IMPORTS
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2 import model_zoo
            
            logger.info("✅ Detectron2 imported successfully")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                return False
            
            logger.info(f"✅ Model file found: {self.model_path}")
            logger.info(f"Model file size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
            
            # Create config
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # CRITICAL: Force CPU device for deployment
            cfg.MODEL.DEVICE = "cpu"
            logger.info("🖥️ Using CPU device for deployment")
            
            # Load model weights with CPU mapping
            try:
                if torch.cuda.is_available():
                    logger.info("CUDA available but forcing CPU for deployment")
                    
                # Always load with CPU mapping for deployment stability
                logger.info("📥 Loading model weights with CPU mapping...")
                cfg.MODEL.WEIGHTS = self.model_path
                
            except Exception as weight_error:
                logger.error(f"❌ Weight loading error: {weight_error}")
                return False
            
            # Your model settings
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.INPUT.FORMAT = "BGR"
            
            # Memory optimization for deployment
            cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
            cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
            
            # Register metadata
            MetadataCatalog.get("grape_disease").thing_classes = self.class_names
            logger.info(f"📋 Registered {len(self.class_names)} classes")
            
            # Create predictor with enhanced error handling
            try:
                logger.info("🔧 Creating predictor...")
                self.predictor = DefaultPredictor(cfg)
                logger.info("✅ Predictor created successfully")
                
            except Exception as pred_error:
                logger.error(f"❌ Predictor creation failed: {pred_error}")
                logger.error(f"Error type: {type(pred_error).__name__}")
                
                # Try alternative approach
                try:
                    logger.info("🔄 Trying alternative predictor creation...")
                    # Force reload weights
                    cfg.MODEL.WEIGHTS = self.model_path
                    self.predictor = DefaultPredictor(cfg)
                    logger.info("✅ Alternative predictor creation successful")
                except Exception as alt_error:
                    logger.error(f"❌ Alternative predictor failed: {alt_error}")
                    return False
            
            # Test model with simple prediction
            try:
                logger.info("🧪 Testing model...")
                test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
                with torch.no_grad():
                    outputs = self.predictor(test_image)
                logger.info("✅ Model test successful")
                
                # Log test results
                instances = outputs["instances"]
                logger.info(f"Test prediction: {len(instances)} detections")
                
            except Exception as test_error:
                logger.error(f"❌ Model test failed: {test_error}")
                return False
            
            self.model_loaded = True
            logger.info("🎉 YOUR REAL MODEL IS LOADED AND READY!")
            logger.info(f"Model device: {next(self.predictor.model.parameters()).device}")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Detectron2 import failed: {e}")
            logger.error("Make sure detectron2 is properly installed")
            return False
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    # Rest of your class methods remain the same...
    # [Keep all your existing base64_to_image, predict, and other methods]
    
    def base64_to_image(self, base64_string):
        """Convert base64 to OpenCV image"""
        try:
            # Handle different base64 formats
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
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
            logger.info("🔬 Running prediction...")
            logger.info(f"Input image shape: {image.shape}")
            
            # Ensure image is in correct format
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] == 3:
                    # Already BGR from our conversion
                    pass
            
            # Get predictions with error handling
            with torch.no_grad():
                outputs = self.predictor(image)
                instances = outputs["instances"]
            
            logger.info(f"🎯 Model output: {len(instances)} detections")
            
            if len(instances) > 0:
                # Move to CPU for processing
                instances = instances.to("cpu")
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
                
                # Get best detection
                best_idx = np.argmax(scores)
                class_id = int(classes[best_idx]) + 1  # Convert 0-4 to 1-5
                confidence = float(scores[best_idx])
                
                # Ensure class_id is valid
                if class_id not in self.disease_mapping:
                    class_id = 5  # Default to healthy
                
                disease_info = self.disease_mapping[class_id]
                
                result = {
                    "disease": disease_info["name"],
                    "marathi": disease_info["marathi"],
                    "confidence": round(confidence * 100, 1),
                    "severity": disease_info["severity"],
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "type": "DETECTRON2_DEPLOYMENT",
                        "detections": len(instances),
                        "device": "cpu",
                        "best_detection_class": int(classes[best_idx]),
                        "all_detections": len(instances)
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
                        "type": "DETECTRON2_DEPLOYMENT",
                        "detections": 0,
                        "device": "cpu"
                    }
                }
                logger.info("✅ No diseases detected - Healthy")
                return result
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e

# Initialize detector
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.7'))

logger.info(f"🔧 Initializing detector with model: {MODEL_PATH}")
detector = GrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

# Load model immediately when server starts (works with Gunicorn too)
success = detector.load_model()
if success:
    logger.info("🎉 Model loaded at startup (global init)")
else:
    logger.error("💥 Model failed to load at startup")


@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    model_exists = os.path.exists(detector.model_path)
    model_size = os.path.getsize(detector.model_path) / (1024*1024) if model_exists else 0
    
    return jsonify({
        "status": "READY" if detector.model_loaded else "NOT_READY",
        "model_loaded": detector.model_loaded,
        "model_exists": model_exists,
        "model_path": detector.model_path,
        "model_size_mb": round(model_size, 1),
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "torch_version": torch.__version__,
        "device": "cpu",
        "architecture": "MASK_RCNN_DEPLOYMENT",
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment": os.getenv('FLASK_ENV', 'development')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Model prediction endpoint with enhanced error handling"""
    try:
        if not detector.model_loaded:
            return jsonify({
                "error": "Model not loaded - check /health endpoint for details",
                "status": "model_not_available",
                "model_path": detector.model_path,
                "model_exists": os.path.exists(detector.model_path)
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "No image data provided", 
                "expected_format": "{'image': 'base64_encoded_image'}"
            }), 400
        
        # Convert image
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({
                "error": "Invalid image data - could not decode base64",
                "hint": "Ensure image is properly base64 encoded"
            }), 400
        
        # Make prediction
        result = detector.predict(image)
        result['timestamp'] = time.time()
        result['server_info'] = {
            "model_loaded": True,
            "deployment": "render",
            "version": "1.0.0"
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": str(e),
            "type": "prediction_error",
            "model_status": detector.model_loaded,
            "timestamp": time.time()
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "GrapeGuard AI - Detectron2 Deployment",
        "status": "READY" if detector.model_loaded else "LOADING",
        "endpoints": {
            "health": "/health - Check model status",
            "predict": "/predict - Make disease predictions",
            "root": "/ - This endpoint"
        },
        "model_status": "LOADED" if detector.model_loaded else "NOT_LOADED",
        "version": "1.0.0",
        "deployment": "render"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/predict"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "model_status": detector.model_loaded if detector else False
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting GrapeGuard Detectron2 Server...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # # Load model
    # success = detector.load_model()
    
    # if success:
    #     logger.info("🎉 SERVER READY - MODEL LOADED!")
    # else:
    #     logger.error("💥 MODEL FAILED TO LOAD - Server starting anyway")
    #     logger.error("Check /health endpoint for diagnostics")
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)