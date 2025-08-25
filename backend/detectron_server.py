# backend/detectron_server.py
import os
import logging
import time
import io
import base64
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load .env from backend folder explicitly
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron_server")

# Config from env
MODEL_PATH_ENV = os.getenv("DETECTRON_MODEL_PATH", "model_final.pth")
# Ensure absolute path if relative
MODEL_PATH = Path(MODEL_PATH_ENV) if Path(MODEL_PATH_ENV).is_absolute() else (BASE_DIR / MODEL_PATH_ENV)
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL")  # optional
THRESHOLD = float(os.getenv("DETECTRON_THRESHOLD", "0.7"))
PORT = int(os.getenv("PORT", "5000"))
CORS_ALLOWED = os.getenv("CORS_ALLOWED_ORIGINS", "*")  # comma-separated or "*" for all

# Flask app
app = Flask(__name__)
origins = [o.strip() for o in CORS_ALLOWED.split(",")] if CORS_ALLOWED != "*" else "*"
CORS(app, resources={r"/*": {"origins": origins}})

# Runtime state
detector = None
model_loaded = False
torch_version = None
detectron2_version = None
detectron2_import_error = None


def ensure_model(path: Path, download_url: Optional[str] = None) -> bool:
    """Return True if model exists and is of reasonable size, otherwise try to download if URL provided."""
    if path.exists() and path.stat().st_size > 100_000_000:  # ~100MB minimal sanity check
        logger.info("Model already present: %s (%.2f MB)", path, path.stat().st_size / (1024 * 1024))
        return True

    if download_url:
        logger.info("Model missing or small. Downloading from %s ...", download_url)
        try:
            r = requests.get(download_url, stream=True, timeout=600)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            logger.info("Model downloaded to %s", path)
            return path.exists() and path.stat().st_size > 100_000_000
        except Exception as e:
            logger.exception("Failed to download model: %s", e)
            return False

    logger.warning("Model not found and no MODEL_DOWNLOAD_URL provided: %s", path)
    return False


class GrapeDiseaseDetector:
    def __init__(self, model_path: Path, threshold: float = 0.7):
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.predictor = None
        self.class_names = [
            "Karpa (Anthracnose)",          # 0
            "Bhuri (Powdery mildew)",       # 1
            "Bokadlela (Borer Infestation)",# 2
            "Davnya (Downy Mildew)",        # 3
            "Healthy"                       # 4
        ]

    def load_model(self) -> bool:
        """Import detectron2 and create DefaultPredictor. Returns True on success."""
        global torch_version, detectron2_version, detectron2_import_error
        try:
            import torch
            torch_version = torch.__version__
            logger.info("torch %s", torch_version)

            # detectron2 imports
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.data import MetadataCatalog

            try:
                import detectron2
                detectron2_version = getattr(detectron2, "__version__", "unknown")
            except Exception:
                detectron2_version = "unknown"

            logger.info("detectron2 import OK (%s)", detectron2_version)

            if not ensure_model(Path(self.model_path), MODEL_DOWNLOAD_URL):
                logger.error("Model not available at %s", self.model_path)
                return False

            cfg = get_cfg()
            # Base COCO Mask R-CNN config; keep this unless you used custom backbone
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.MODEL.DEVICE = "cpu"
            cfg.DATALOADER.NUM_WORKERS = 0

            MetadataCatalog.get("grape_disease").thing_classes = self.class_names

            logger.info("Creating DefaultPredictor (this may take a moment)...")
            self.predictor = DefaultPredictor(cfg)
            logger.info("Predictor created.")

            # quick sanity inference
            import numpy as np
            test_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
            _ = self.predictor(test_img)
            logger.info("Sanity inference OK.")
            return True
        except ImportError as e:
            detectron2_import_error = str(e)
            logger.exception("Detectron2 import failed: %s", e)
            return False
        except Exception as e:
            logger.exception("Model loading failed: %s", e)
            return False

    def predict_from_image(self, image_bgr):
        """Run prediction and return dict with structured results and visualization (base64)."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        outputs = self.predictor(image_bgr)
        instances = outputs.get("instances", None)
        results = {
            "predictions": [],
            "processing_time": None,
            "visualization": None,
            "detections": 0
        }

        if instances is None or len(instances) == 0:
            results["processing_time"] = round(time.time() - start, 3)
            results["detections"] = 0
            return results

        # Move instances to cpu for safe indexing
        inst_cpu = instances.to("cpu")
        scores = inst_cpu.scores.tolist() if hasattr(inst_cpu, "scores") else []
        classes = inst_cpu.pred_classes.tolist() if hasattr(inst_cpu, "pred_classes") else []
        boxes = (inst_cpu.pred_boxes.tensor.tolist() if hasattr(inst_cpu, "pred_boxes") else [])

        for i in range(len(scores)):
            cls = int(classes[i])  # 0-based
            score = float(scores[i])
            bbox = boxes[i] if i < len(boxes) else None
            results["predictions"].append({
                "class_id": cls + 1,  # keep 1..5 mapping for frontend compatibility
                "class_name": self.class_names[cls] if cls < len(self.class_names) else "Unknown",
                "confidence": round(score, 4),
                "bbox": bbox
            })

        # Visualization: draw predictions onto image
        try:
            from detectron2.utils.visualizer import Visualizer
            from detectron2.data import MetadataCatalog
            vis = Visualizer(image_bgr[:, :, ::-1], MetadataCatalog.get("grape_disease"), scale=1.0)
            vis_out = vis.draw_instance_predictions(inst_cpu)
            vis_img = vis_out.get_image()[:, :, ::-1]  # RGB -> BGR
            import cv2
            success, encoded = cv2.imencode('.jpg', vis_img)
            if success:
                results["visualization"] = base64.b64encode(encoded.tobytes()).decode("utf-8")
        except Exception as e:
            logger.warning("Visualization failed: %s", e)

        results["processing_time"] = round(time.time() - start, 3)
        results["detections"] = len(results["predictions"])
        return results


# Create detector instance (but do not fail import if loading fails)
detector = GrapeDiseaseDetector(MODEL_PATH, THRESHOLD)

# Attempt model loading at import time (so Gunicorn/Render import triggers it)
try:
    logger.info("Attempting to load model at import time...")
    loaded_ok = detector.load_model()
    if loaded_ok:
        model_loaded = True
        logger.info("Model loaded successfully at import time.")
    else:
        model_loaded = False
        logger.warning("Model did NOT load at import time.")
except Exception as e:
    model_loaded = False
    logger.exception("Model load threw exception at import time: %s", e)


# Helpers: convert multipart file / base64 / imageUrl to OpenCV (BGR) np array
def pil_file_to_bgr(file_stream) -> Optional[object]:
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(file_stream)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        return arr
    except Exception as e:
        logger.exception("Failed to convert PIL->BGR: %s", e)
        return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "READY" if model_loaded else "NOT_READY",
        "model_loaded": model_loaded,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
        "threshold": THRESHOLD,
        "torch_version": torch_version,
        "detectron2_version": detectron2_version,
        "detectron2_import_error": detectron2_import_error,
        "classes": detector.class_names
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    if not model_loaded:
        return jsonify({"error": "Model not loaded", "model_loaded": False}), 503

    # Accept: multipart (file), JSON: { image: "<base64>", imageUrl: "<url>" }
    image_bgr = None

    # multipart file upload
    if 'image' in request.files:
        file = request.files['image'].read()
        image_bgr = pil_file_to_bgr(file)

    # JSON body
    if image_bgr is None:
        data = None
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if data:
            if "image" in data:  # base64 string (may include data: prefix)
                b = data["image"]
                if b.startswith("data:"):
                    b = b.split(",", 1)[1]
                try:
                    decoded = base64.b64decode(b)
                    image_bgr = pil_file_to_bgr(decoded)
                except Exception as e:
                    logger.exception("Failed to decode base64 image: %s", e)
                    return jsonify({"error": "Invalid base64 image"}), 400

            elif "imageUrl" in data:
                try:
                    r = requests.get(data["imageUrl"], stream=True, timeout=30)
                    r.raise_for_status()
                    image_bgr = pil_file_to_bgr(r.content)
                except Exception as e:
                    logger.exception("Failed to download imageUrl: %s", e)
                    return jsonify({"error": "Failed to download imageUrl", "detail": str(e)}), 400

    if image_bgr is None:
        return jsonify({"error": "No image provided. Send multipart-file (image) or JSON {image: base64} or {imageUrl}"}), 400

    try:
        res = detector.predict_from_image(image_bgr)
        # map class ids to your marathi/severity mapping if you want (frontend already has mapping)
        res["timestamp"] = time.time()
        res["model_info"] = {"type": "detectron2", "torch_version": torch_version, "detectron2_version": detectron2_version}
        return jsonify(res)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({"error": "prediction_failed", "detail": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "GrapeGuard Detectron2 server",
        "status": "READY" if model_loaded else "NOT_READY",
        "endpoints": ["/health", "/predict"]
    })


if __name__ == "__main__":
    # When running directly, try loading model if not loaded and start Flask dev server (for local dev)
    if not model_loaded:
        try:
            logger.info("Loading model in __main__ before starting local server...")
            model_loaded = detector.load_model()
        except Exception as e:
            logger.exception("Load in __main__ failed: %s", e)

    logger.info("Starting local Flask server on 0.0.0.0:%s ...", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)
