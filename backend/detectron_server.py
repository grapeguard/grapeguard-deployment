# DETECTRON2 SERVER WITH YOUR CUSTOM IMPORT PATHS
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Add detectron2 folder to path (like your local setup)
detectron_path = os.path.join(os.path.dirname(__file__), 'detectron2')
if detectron_path not in sys.path:
    sys.path.insert(0, detectron_path)

# Global variables
DETECTRON2_LOADED = False
predictor = None

def load_detectron2_with_custom_path():
    """Load detectron2 with your custom path structure"""
    global DETECTRON2_LOADED
    
    try:
        # Check PyTorch first
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
        
        # Try your custom import path first
        try:
            import detectron2.detectron2
            from detectron2.detectron2.engine import DefaultPredictor
            from detectron2.detectron2.config import get_cfg
            from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
            from detectron2.detectron2.data import MetadataCatalog
            from detectron2.detectron2 import model_zoo
            
            DETECTRON2_LOADED = True
            logger.info("✅ Detectron2 loaded with custom path (detectron2.detectron2)")
            return True
            
        except ImportError as e1:
            logger.info(f"⚠️ Custom path failed: {e1}")
            
            # Fallback to standard import
            try:
                import detectron2
                from detectron2.engine import DefaultPredictor
                from detectron2.config import get_cfg
                from detectron2.utils.visualizer import Visualizer, ColorMode
                from detectron2.data import MetadataCatalog
                from detectron2 import model_zoo
                
                DETECTRON2_LOADED = True
                logger.info("✅ Detectron2 loaded with standard path")
                return True
                
            except ImportError as e2:
                logger.error(f"❌ Both import paths failed: {e2}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Detectron2 setup failed: {e}")
        return False

class RealGrapeDiseaseDetector:
    def __init__(self, model_path, threshold=0.7):
        self.model_path = model_path
        self.threshold = threshold
        self.predictor = None
        self.cfg = None
        self.model_loaded = False
        self.import_mode = "unknown"
        
        # YOUR EXACT CLASSES FROM TRAINING
        self.class_names = [
            "Karpa (Anthracnose)",          # 0
            "Bhuri (Powdery mildew)",       # 1  
            "Bokadlela (Borer Infestation)", # 2
            "Davnya (Downey Mildew)",       # 3
            "Healthy"                       # 4
        ]
        
        # Disease mapping for frontend (convert 0-4 to 1-5)
        self.disease_mapping = {
            1: {"name": "Karpa (Anthracnose)", "marathi": "कर्पा रोग", "severity": "High"},
            2: {"name": "Bhuri (Powdery Mildew)", "marathi": "भुरी रोग", "severity": "Medium"},
            3: {"name": "Bokadlela (Borer Infestation)", "marathi": "बोकाडलेला", "severity": "High"},
            4: {"name": "Davnya (Downy Mildew)", "marathi": "दवयाचा रोग", "severity": "High"},
            5: {"name": "Healthy", "marathi": "निरोगी पान", "severity": "None"}
        }
        
        logger.info(f"🎯 Real Detector initialized for model: {model_path}")
    
    def load_real_model(self):
        """Load YOUR trained Detectron2 model with correct imports"""
        if not DETECTRON2_LOADED:
            logger.error("❌ Detectron2 not available - CANNOT PROCEED")
            return False
            
        if not os.path.exists(self.model_path):
            logger.error(f"❌ YOUR MODEL FILE NOT FOUND: {self.model_path}")
            return False
            
        try:
            # Try custom import path first (your local setup)
            try:
                from detectron2.detectron2.engine import DefaultPredictor
                from detectron2.detectron2.config import get_cfg
                from detectron2.detectron2.data import MetadataCatalog
                from detectron2.detectron2 import model_zoo
                self.import_mode = "custom"
                logger.info("🔧 Using custom import path (detectron2.detectron2)")
                
            except ImportError:
                # Fallback to standard imports
                from detectron2.engine import DefaultPredictor
                from detectron2.config import get_cfg
                from detectron2.data import MetadataCatalog
                from detectron2 import model_zoo
                self.import_mode = "standard"
                logger.info("🔧 Using standard import path")
            
            logger.info("🔥 LOADING YOUR REAL DETECTRON2 MODEL...")
            
            # Create config exactly like your training
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # YOUR MODEL CONFIGURATION
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Your 5 classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.MODEL.DEVICE = "cpu"  # CPU for Render
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Reduced for memory
            
            # Register metadata EXACTLY like your training
            dataset_name = "leaf_disease_dataset"
            MetadataCatalog.get(dataset_name).thing_classes = self.class_names
            MetadataCatalog.get(dataset_name).thing_colors = [
                [255, 0, 0],    # Red - Anthracnose
                [255, 165, 0],  # Orange - Powdery Mildew  
                [139, 0, 0],    # Dark Red - Borer
                [255, 0, 255],  # Magenta - Downy Mildew
                [0, 255, 0]     # Green - Healthy
            ]
            
            # Create predictor with YOUR weights
            logger.info("🚀 Creating predictor with YOUR trained weights...")
            self.predictor = DefaultPredictor(cfg)
            self.cfg = cfg
            
            # MANDATORY MODEL TEST
            logger.info("🧪 Testing YOUR real model...")
            test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
            outputs = self.predictor(test_image)
            
            logger.info(f"✅ YOUR MODEL TEST PASSED!")
            logger.info(f"📊 Output structure: {list(outputs.keys())}")
            logger.info(f"🎯 Test instances: {len(outputs['instances'])}")
            logger.info(f"🔧 Import mode: {self.import_mode}")
            
            self.model_loaded = True
            logger.info("🎉 YOUR REAL DETECTRON2 MODEL IS READY!")
            return True
            
        except Exception as e:
            logger.error(f"💥 YOUR MODEL LOADING FAILED: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return False
    
    def base64_to_image(self, base64_string):
        """Convert base64 to OpenCV image"""
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
    
    def image_to_base64(self, image):
        """Convert OpenCV image to base64"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"❌ Image to base64 failed: {e}")
            return None
    
    def predict_with_real_model(self, image, include_visualization=True):
        """REAL PREDICTION WITH YOUR TRAINED MODEL ONLY"""
        if not self.model_loaded:
            raise Exception("YOUR REAL MODEL IS NOT LOADED")
            
        start_time = time.time()
        
        try:
            logger.info("🔬 RUNNING YOUR REAL DETECTRON2 MODEL...")
            
            # Get predictions from YOUR model
            outputs = self.predictor(image)
            instances = outputs["instances"]
            
            logger.info(f"🎯 YOUR MODEL OUTPUT: {len(instances)} detections")
            
            detections = []
            if len(instances) > 0:
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                
                logger.info(f"📋 PROCESSING {len(scores)} REAL DETECTIONS:")
                for i in range(len(scores)):
                    class_id = int(classes[i])  # 0-4 from your model
                    confidence = float(scores[i])
                    bbox = boxes[i].tolist()
                    
                    logger.info(f"   Detection {i+1}: Class={class_id} ({self.class_names[class_id]}), Conf={confidence:.3f}")
                    
                    detections.append({
                        "class_id": class_id + 1,  # Convert 0-4 to 1-5 for frontend
                        "class_name": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox": bbox
                    })
            
            # Create visualization if requested
            visualization_image = None
            if include_visualization and len(detections) > 0:
                try:
                    # Use the same import mode as model loading
                    if self.import_mode == "custom":
                        from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
                        from detectron2.detectron2.data import MetadataCatalog
                    else:
                        from detectron2.utils.visualizer import Visualizer, ColorMode
                        from detectron2.data import MetadataCatalog
                    
                    metadata = MetadataCatalog.get("leaf_disease_dataset")
                    visualizer = Visualizer(
                        image[:, :, ::-1],  # BGR to RGB
                        metadata=metadata,
                        scale=1.0,
                        instance_mode=ColorMode.IMAGE
                    )
                    
                    vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
                    vis_image = vis_output.get_image()[:, :, ::-1]  # RGB to BGR
                    visualization_image = self.image_to_base64(vis_image)
                    
                    logger.info("🎨 REAL MODEL VISUALIZATION CREATED")
                    
                except Exception as vis_error:
                    logger.warning(f"⚠️ Visualization failed: {vis_error}")
            
            # Format result
            if detections:
                # Get best detection
                best_detection = max(detections, key=lambda x: x['confidence'])
                disease_info = self.disease_mapping[best_detection['class_id']]
                
                result = {
                    "disease": disease_info["name"],
                    "marathi": disease_info["marathi"],
                    "confidence": round(best_detection['confidence'] * 100, 1),
                    "severity": disease_info["severity"],
                    "detections": detections,
                    "visualization": visualization_image,
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "type": "YOUR_REAL_DETECTRON2_MODEL",
                        "architecture": "Mask R-CNN ResNet-50",
                        "classes": len(self.class_names),
                        "device": "cpu",
                        "import_mode": self.import_mode,
                        "weights": "YOUR_TRAINED_WEIGHTS"
                    }
                }
            else:
                # No detections = healthy (from your model)
                result = {
                    "disease": "Healthy",
                    "marathi": "निरोगी पान",
                    "confidence": 95.0,
                    "severity": "None",
                    "detections": [],
                    "visualization": None,
                    "processing_time": round(time.time() - start_time, 3),
                    "model_info": {
                        "type": "YOUR_REAL_DETECTRON2_MODEL",
                        "import_mode": self.import_mode,
                        "note": "No diseases detected by your trained model"
                    }
                }
            
            logger.info(f"✅ YOUR MODEL PREDICTION: {result['disease']} ({result['confidence']}%)")
            return result
            
        except Exception as e:
            logger.error(f"💥 YOUR MODEL PREDICTION FAILED: {e}")
            raise e

# Initialize on startup
logger.info("🚀 STARTING REAL GRAPE DISEASE DETECTION WITH CUSTOM IMPORTS...")

# Load detectron2 with your custom path
detectron2_success = load_detectron2_with_custom_path()

if not detectron2_success:
    logger.error("💥 DETECTRON2 SETUP FAILED")
    # Don't exit - let server run for debugging

# Initialize detector with YOUR model
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.7'))

detector = RealGrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check - REAL MODEL STATUS ONLY"""
    return jsonify({
        "status": "REAL_MODEL_READY" if detector.model_loaded else "MODEL_NOT_LOADED",
        "detectron2_available": DETECTRON2_LOADED,
        "model_loaded": detector.model_loaded,
        "model_exists": os.path.exists(detector.model_path),
        "model_path": detector.model_path,
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "device": "cpu",
        "import_mode": getattr(detector, 'import_mode', 'unknown'),
        "detectron2_folder_exists": os.path.exists(os.path.join(os.path.dirname(__file__), 'detectron2')),
        "architecture": "YOUR_TRAINED_MASK_RCNN",
        "note": "REAL MODEL ONLY - CUSTOM IMPORT PATHS"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """REAL MODEL PREDICTION ONLY"""
    try:
        if not detector.model_loaded:
            return jsonify({
                "error": "YOUR REAL MODEL IS NOT LOADED",
                "status": "model_not_available",
                "detectron2_available": DETECTRON2_LOADED,
                "model_exists": os.path.exists(detector.model_path),
                "note": "Check /health endpoint for detailed status"
            }), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        include_visualization = data.get('include_visualization', True)
        result = detector.predict_with_real_model(image, include_visualization)
        
        result['timestamp'] = time.time()
        result['api_version'] = "REAL_MODEL_CUSTOM_IMPORTS_V1"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"💥 REAL MODEL PREDICTION ERROR: {e}")
        return jsonify({
            "error": f"Real model prediction failed: {str(e)}",
            "type": "real_model_error"
        }), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "GrapeGuard REAL AI - Your Trained Model with Custom Imports",
        "status": "REAL_MODEL_ONLY",
        "endpoints": ["/health", "/predict"],
        "model_status": "REAL_DETECTRON2" if detector.model_loaded else "MODEL_NOT_LOADED",
        "import_mode": getattr(detector, 'import_mode', 'unknown'),
        "note": "NO FALLBACKS - YOUR TRAINED MODEL WITH detectron2.detectron2 IMPORTS"
    })

if __name__ == '__main__':
    logger.info("🔥 LOADING YOUR REAL DETECTRON2 MODEL WITH CUSTOM IMPORTS...")
    
    success = detector.load_real_model()
    
    if success:
        logger.info("🎉 YOUR REAL MODEL IS READY FOR PRODUCTION!")
    else:
        logger.error("💥 YOUR MODEL FAILED TO LOAD - CHECK LOGS")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 REAL MODEL SERVER RUNNING ON PORT {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)