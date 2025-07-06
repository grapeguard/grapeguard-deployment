# Production-Ready Detectron2 Server for Render Deployment
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["https://your-frontend-domain.vercel.app", "http://localhost:3000"])

# Add detectron2 to path for your custom installation
detectron_path = os.path.join(os.path.dirname(__file__), 'detectron')
if detectron_path not in sys.path:
    sys.path.insert(0, detectron_path)

# Import detectron2 with your custom path
try:
    import detectron2.detectron2
    from detectron2.detectron2.engine import DefaultPredictor
    from detectron2.detectron2.config import get_cfg
    from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.detectron2.data import MetadataCatalog
    from detectron2.detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
    logger.info("✅ Detectron2 imported successfully")
except ImportError as e:
    logger.error(f"❌ Detectron2 import failed: {e}")
    DETECTRON2_AVAILABLE = False

class GrapeDiseaseDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.predictor = None
        self.detectron2_loaded = False
        
        # EXACT same classes as your training
        self.class_names = [
            "Karpa (Anthracnose)",          # Class 0
            "Bhuri (Powdery mildew)",       # Class 1  
            "Bokadlela (Borer Infestation)", # Class 2
            "Davnya (Downey Mildew)",       # Class 3
            "Healthy"                       # Class 4
        ]
        
        # Colors for visualization (RGB format)
        self.colors = [
            [255, 0, 0],    # Red for Anthracnose
            [255, 165, 0],  # Orange for Powdery Mildew
            [139, 0, 0],    # Dark Red for Borer
            [255, 0, 255],  # Magenta for Downy Mildew
            [0, 255, 0]     # Green for Healthy
        ]
        
        logger.info(f"🔧 Model path: {model_path}")
        logger.info(f"🎯 Detection threshold: {threshold}")
        logger.info(f"🏷️ Classes: {self.class_names}")
        
    def download_model_if_needed(self):
        """Download model from external storage if not present"""
        if not os.path.exists(self.model_path):
            model_url = os.getenv('MODEL_DOWNLOAD_URL')
            if model_url:
                logger.info(f"📥 Downloading model from {model_url}")
                try:
                    import requests
                    response = requests.get(model_url, stream=True)
                    response.raise_for_status()
                    
                    with open(self.model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info("✅ Model downloaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"❌ Model download failed: {e}")
                    return False
            else:
                logger.error("❌ Model not found and no download URL provided")
                return False
        return True
        
    def create_config_like_training(self):
        """Create config identical to your training setup"""
        try:
            cfg = get_cfg()
            
            # Try to find config file in your detectron folder
            config_path = None
            possible_paths = [
                "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                "detectron/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                "./detectron/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path:
                cfg.merge_from_file(config_path)
                logger.info(f"✅ Loaded config from: {config_path}")
            else:
                # Manual config setup for production
                logger.info("⚙️ Using manual config setup for production")
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
                cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
                cfg.MODEL.RESNETS.DEPTH = 50
                cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
                cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
                cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
                cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
                cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
                cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
                cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
                cfg.INPUT.MIN_SIZE_TRAIN = (800,)
                cfg.INPUT.MAX_SIZE_TRAIN = 1333
                cfg.INPUT.MIN_SIZE_TEST = 800
                cfg.INPUT.MAX_SIZE_TEST = 1333
            
            # Your trained model weights
            cfg.MODEL.WEIGHTS = self.model_path
            
            # Same settings as training
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            
            # Device settings - Force CPU for deployment reliability
            cfg.MODEL.DEVICE = "cpu"  # Use CPU for consistent deployment
            
            # Data settings
            cfg.DATALOADER.NUM_WORKERS = 0
            
            logger.info("✅ Config created for production deployment")
            return cfg, True
            
        except Exception as e:
            logger.error(f"❌ Config creation failed: {e}")
            return None, False
    
    def load_model(self):
        """Load your trained model with correct config"""
        if not DETECTRON2_AVAILABLE:
            logger.error("❌ Detectron2 not available")
            return False
            
        # Download model if needed
        if not self.download_model_if_needed():
            logger.error("❌ Model file not available")
            return False
            
        try:
            logger.info("🔥 LOADING YOUR TRAINED MODEL FOR PRODUCTION...")
            
            # Create config matching your training
            cfg, success = self.create_config_like_training()
            if not success:
                logger.error("❌ Failed to create config")
                return False
            
            # Register metadata exactly like your training
            dataset_name = "leaf_disease_dataset"
            MetadataCatalog.get(dataset_name).thing_classes = self.class_names
            MetadataCatalog.get(dataset_name).thing_colors = self.colors
            
            logger.info(f"✅ Metadata registered for dataset: {dataset_name}")
            
            # Create predictor
            logger.info("🚀 Creating predictor for production...")
            self.predictor = DefaultPredictor(cfg)
            self.cfg = cfg
            
            # Test the model with smaller image for production
            logger.info("🧪 Testing your model in production environment...")
            test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
            outputs = self.predictor(test_image)
            
            logger.info(f"✅ PRODUCTION MODEL TEST SUCCESSFUL!")
            logger.info(f"🎯 Test output instances: {len(outputs['instances'])}")
            
            self.detectron2_loaded = True
            logger.info("🎉 YOUR TRAINED MODEL IS READY FOR PRODUCTION!")
            return True
            
        except Exception as e:
            logger.error(f"💥 PRODUCTION MODEL LOADING FAILED: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            logger.error(f"❌ Image conversion failed: {str(e)}")
            return None
    
    def image_to_base64(self, image):
        """Convert OpenCV image to base64"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)  # Slightly lower quality for production
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"❌ Image to base64 conversion failed: {str(e)}")
            return None
    
    def predict(self, image, include_visualization=True):
        """Make prediction using your trained model"""
        if not self.detectron2_loaded:
            return {
                "predictions": [],
                "visualization": None,
                "error": "Your trained model is not loaded",
                "model_info": {"architecture": "MODEL NOT LOADED"}
            }
        
        start_time = time.time()
        
        try:
            logger.info("🔬 Running production prediction...")
            
            # Get predictions from your model
            outputs = self.predictor(image)
            instances = outputs["instances"]
            
            logger.info(f"🎯 Production output: {len(instances)} instances detected")
            
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
                        "class_id": class_id + 1,  # Convert 0-4 to 1-5 for frontend
                        "class_name": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox": bbox
                    })
            
            # Create visualization (optional in production for performance)
            visualization_image = None
            if include_visualization and len(detections) > 0:
                try:
                    metadata = MetadataCatalog.get("leaf_disease_dataset")
                    visualizer = Visualizer(
                        image[:, :, ::-1],
                        metadata=metadata,
                        scale=0.8,  # Smaller scale for production
                        instance_mode=ColorMode.IMAGE
                    )
                    
                    vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
                    vis_image = vis_output.get_image()[:, :, ::-1]
                    visualization_image = self.image_to_base64(vis_image)
                    
                except Exception as vis_error:
                    logger.warning(f"⚠️ Visualization creation failed: {vis_error}")
                    visualization_image = None
            
            processing_time = time.time() - start_time
            
            result = {
                "predictions": detections,
                "visualization": visualization_image,
                "image_info": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "size": f"{image.shape[1]}x{image.shape[0]}"
                },
                "processing_time": round(processing_time, 3),
                "model_info": {
                    "architecture": "YOUR TRAINED MASK R-CNN (ResNet-50)",
                    "device": "cpu",
                    "num_classes": 5,
                    "threshold": self.threshold,
                    "environment": "production"
                }
            }
            
            logger.info(f"🎉 PRODUCTION PREDICTION COMPLETE! ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"💥 PRODUCTION PREDICTION FAILED: {str(e)}")
            
            return {
                "predictions": [],
                "visualization": None,
                "error": f"Production prediction failed: {str(e)}",
                "model_info": {"architecture": "PRODUCTION ERROR", "error": str(e)}
            }

# Initialize detector
MODEL_PATH = os.getenv('DETECTRON_MODEL_PATH', './model_final.pth')
THRESHOLD = float(os.getenv('DETECTRON_THRESHOLD', '0.5'))

detector = GrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if detector.detectron2_loaded else "model_not_loaded",
        "detectron2_available": DETECTRON2_AVAILABLE,
        "detectron2_loaded": detector.detectron2_loaded,
        "model_exists": os.path.exists(detector.model_path),
        "model_path": detector.model_path,
        "device": "cpu",
        "classes": detector.class_names,
        "threshold": detector.threshold,
        "environment": "production"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Production prediction endpoint"""
    try:
        if not detector.detectron2_loaded:
            return jsonify({
                "error": "Model not loaded. Check server logs.",
                "status": "model_not_available"
            }), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image = detector.base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        include_visualization = data.get('include_visualization', False)  # Default false for production performance
        result = detector.predict(image, include_visualization)
        
        result['timestamp'] = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Production endpoint error: {str(e)}")
        return jsonify({
            "error": str(e),
            "type": "production_error"
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get production model information"""
    return jsonify({
        "model_path": detector.model_path,
        "model_exists": os.path.exists(detector.model_path),
        "threshold": detector.threshold,
        "classes": detector.class_names,
        "device": "cpu",
        "detectron2_loaded": detector.detectron2_loaded,
        "architecture": "Mask R-CNN with ResNet-50 FPN",
        "environment": "production",
        "deployment": "render",
        "note": "Production-optimized configuration"
    })

if __name__ == '__main__':
    logger.info("🚀 STARTING PRODUCTION GRAPE DISEASE DETECTION")
    
    success = detector.load_model()
    
    if success:
        logger.info("🎉 PRODUCTION MODEL READY!")
    else:
        logger.error("💥 PRODUCTION MODEL FAILED TO LOAD!")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Starting production server on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)