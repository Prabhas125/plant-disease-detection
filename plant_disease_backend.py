"""
PlantGuard AI — Plant Disease Detection Backend
================================================

A small Flask app that serves the website and runs inference with the
EfficientNet-B0 model trained in ``plant_disease_training_colab.ipynb``.

The model is trained on the public PlantVillage subset
``GVJahnavi/Plant_village_subset``:
    14,238 images · 15 classes · 5 crops (Apple, Blueberry, Cherry, Corn, Grape)
    Checkpoint: ~21 MB

SETUP
-----
1. Run the notebook and download ``plant_disease_model.pth``.
2. Put it here:   models/plant_disease_model.pth
3. Install dependencies:
       pip install flask pillow torch torchvision
4. Start the server:
       python plant_disease_backend.py
5. Open http://127.0.0.1:5000

The class list is read from the checkpoint, so the backend and the model
can never fall out of sync.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory

# ── Paths & constants ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "plant_disease_model.pth"
WEBSITE = BASE_DIR / "plant_disease_website.html"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_MB = 10
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("plantguard")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Disease knowledge base ───────────────────────────────────────────────────
# Keys must match the class names saved in the checkpoint exactly.
CLASS_INFO = {
    "Apple__Apple_scab": {
        "plant": "Apple", "disease": "Apple Scab", "severity": "high",
        "description": "Fungal disease (Venturia inaequalis) causing dark, scabby "
                       "olive-green spots on leaves and fruit.",
        "treatment": "Apply captan or myclobutanil fungicide. Rake up and remove "
                     "fallen leaves. Prune for better airflow.",
        "symptoms": ["Dark olive-green spots", "Distorted leaves", "Cracked fruit",
                     "Premature leaf drop"],
    },
    "Apple__Black_rot": {
        "plant": "Apple", "disease": "Black Rot", "severity": "high",
        "description": "Fungal infection causing black, rotting lesions on fruit and "
                       "cankers on branches.",
        "treatment": "Prune and destroy infected branches. Apply thiophanate-methyl "
                     "fungicide. Remove mummified fruit.",
        "symptoms": ["Black circular lesions", "Mummified fruit", "Bark cankers",
                     "Purple-edged leaf spots"],
    },
    "Apple__Cedar_apple_rust": {
        "plant": "Apple", "disease": "Cedar Apple Rust", "severity": "medium",
        "description": "Rust disease that needs two hosts: apple trees and nearby "
                       "cedar or juniper trees.",
        "treatment": "Apply myclobutanil or mancozeb in spring. Remove nearby "
                     "juniper hosts where practical.",
        "symptoms": ["Bright orange-yellow spots", "Yellow halos", "Distorted fruit",
                     "Small raised lesions"],
    },
    "Apple___healthy": {
        "plant": "Apple", "disease": "Healthy", "severity": "none",
        "description": "No disease detected. The leaf appears healthy.",
        "treatment": "Continue regular monitoring, pruning and balanced fertilisation.",
        "symptoms": [],
    },
    "Blueberry___healthy": {
        "plant": "Blueberry", "disease": "Healthy", "severity": "none",
        "description": "No disease detected. The leaf appears healthy.",
        "treatment": "Maintain soil pH around 4.5–5.5 and keep plants well watered.",
        "symptoms": [],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry", "disease": "Powdery Mildew", "severity": "medium",
        "description": "Fungal disease producing a white, powdery coating on leaves "
                       "and young shoots.",
        "treatment": "Apply a sulfur-based fungicide. Improve air circulation by "
                     "pruning. Avoid excess nitrogen.",
        "symptoms": ["White powdery patches", "Stunted growth", "Leaf curling",
                     "Distorted new shoots"],
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry", "disease": "Healthy", "severity": "none",
        "description": "No disease detected. The leaf appears healthy.",
        "treatment": "Prune annually for airflow and monitor for aphids.",
        "symptoms": [],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)", "disease": "Gray Leaf Spot", "severity": "high",
        "description": "Fungal disease causing long, grey rectangular lesions that "
                       "reduce photosynthesis and yield.",
        "treatment": "Use resistant hybrids and rotate crops. Apply azoxystrobin "
                     "fungicide when symptoms first appear.",
        "symptoms": ["Grey rectangular lesions", "Tan centre with dark border",
                     "Lesions run parallel to veins", "Premature leaf death"],
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)", "disease": "Common Rust", "severity": "medium",
        "description": "Rust disease producing cinnamon-brown pustules on both "
                       "surfaces of the leaf.",
        "treatment": "Apply propiconazole fungicide if severe. Plant resistant "
                     "varieties and avoid late planting.",
        "symptoms": ["Cinnamon-brown pustules", "Pustules on both leaf surfaces",
                     "Yellow halo around pustules"],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)", "disease": "Northern Leaf Blight", "severity": "high",
        "description": "Fungal disease causing long, cigar-shaped grey-green to tan "
                       "lesions, often starting on lower leaves.",
        "treatment": "Apply a strobilurin fungicide early. Rotate crops and use "
                     "resistant hybrids.",
        "symptoms": ["Long cigar-shaped lesions (6–15 cm)", "Grey-green to tan colour",
                     "Lower leaves affected first", "Premature leaf death"],
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)", "disease": "Healthy", "severity": "none",
        "description": "No disease detected. The leaf appears healthy.",
        "treatment": "Maintain balanced N-P-K fertilisation and monitor weekly.",
        "symptoms": [],
    },
    "Grape___Black_rot": {
        "plant": "Grape", "disease": "Black Rot", "severity": "high",
        "description": "Devastating fungal disease that shrivels grapes into hard "
                       "black mummies and causes brown leaf spots.",
        "treatment": "Apply mancozeb or myclobutanil. Remove and destroy mummified "
                     "fruit and infected canes.",
        "symptoms": ["Brown circular leaf spots", "Black shrivelled fruit",
                     "Dark cankers on shoots"],
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape", "disease": "Esca (Black Measles)", "severity": "high",
        "description": "Complex wood disease producing tiger-stripe leaf symptoms "
                       "and berry shrivel. There is no chemical cure.",
        "treatment": "Remove severely infected vines. Use trunk renewal and avoid "
                     "pruning in wet weather.",
        "symptoms": ["Tiger-stripe leaves", "Berry shrivel", "Dark wood discolouration"],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape", "disease": "Leaf Blight", "severity": "medium",
        "description": "Fungal leaf disease producing dark brown angular spots that "
                       "can lead to early defoliation.",
        "treatment": "Apply a copper-based fungicide. Improve canopy ventilation "
                     "and remove infected leaves.",
        "symptoms": ["Dark brown angular spots", "Yellow margins",
                     "Premature defoliation"],
    },
    "Grape___healthy": {
        "plant": "Grape", "disease": "Healthy", "severity": "none",
        "description": "No disease detected. The leaf appears healthy.",
        "treatment": "Monitor weekly for powdery mildew and keep the canopy open.",
        "symptoms": [],
    },
}


def describe(class_name: str) -> dict:
    """Return metadata for a class, with a safe fallback for unknown names."""
    if class_name in CLASS_INFO:
        return CLASS_INFO[class_name]

    # Fallback: "Plant___Disease_name" -> readable text
    if "___" in class_name:
        plant_raw, disease_raw = class_name.split("___", 1)
    else:
        plant_raw, disease_raw = class_name, "Unknown"

    plant = plant_raw.replace("_", " ").replace("(including sour)", "").strip()
    disease = disease_raw.replace("_", " ").strip()
    healthy = "healthy" in disease.lower()

    return {
        "plant": plant,
        "disease": "Healthy" if healthy else disease.title(),
        "severity": "none" if healthy else "medium",
        "description": "No disease detected." if healthy
                       else f"{disease.title()} detected by the model.",
        "treatment": "Continue regular monitoring." if healthy
                     else "Consult your local agricultural officer for treatment advice.",
        "symptoms": [],
    }


# ── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """Must stay identical to build_model() in the training notebook."""
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)


class Predictor:
    """Loads the checkpoint and runs inference."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names: list[str] = []
        self.meta: dict = {}
        self.error: str | None = None

        if not self.model_path.is_file():
            self.error = (
                f"Model file not found: {self.model_path}\n"
                "Run the notebook, download plant_disease_model.pth and place it "
                "in the models/ folder."
            )
            log.warning(self.error)
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
            self.class_names = list(checkpoint["class_names"])
            self.meta = {
                "architecture": checkpoint.get("architecture", "efficientnet_b0"),
                "val_acc": checkpoint.get("val_acc"),
                "test_accuracy": checkpoint.get("test_accuracy"),
                "macro_f1": checkpoint.get("macro_f1"),
                "best_epoch": checkpoint.get("best_epoch"),
                "dataset": checkpoint.get("dataset"),
                "train_images": checkpoint.get("train_images"),
                "val_images": checkpoint.get("val_images"),
                "trained_at": checkpoint.get("trained_at"),
                "history": checkpoint.get("history", {}),
            }

            model = build_model(len(self.class_names))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval().to(DEVICE)
            self.model = model

            log.info("Loaded %s (%d classes, val acc %s%% ) from %s",
                     self.meta["architecture"], len(self.class_names),
                     self.meta["val_acc"], self.model_path.name)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Could not load the model: {exc}"
            log.exception("Model loading failed")

    @property
    def ready(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "Model is not loaded")

        image = Image.open(io.BytesIO(image_bytes))
        tensor = preprocess(image)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        k = min(5, len(self.class_names))
        top_probs, top_idx = torch.topk(probs, k=k)

        top = []
        for idx, prob in zip(top_idx.tolist(), top_probs.tolist()):
            name = self.class_names[idx]
            info = describe(name)
            top.append({
                "class": name,
                "plant": info["plant"],
                "disease": info["disease"],
                "confidence": round(prob * 100, 2),
            })

        best = top[0]
        info = describe(best["class"])

        return {
            "predicted_class": best["class"],
            "plant": info["plant"],
            "disease": info["disease"],
            "confidence": best["confidence"],
            "severity": info["severity"],
            "description": info["description"],
            "treatment": info["treatment"],
            "symptoms": info["symptoms"],
            "is_healthy": info["disease"] == "Healthy",
            "top5": top,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


# ── Flask app ────────────────────────────────────────────────────────────────
def create_app(predictor: Predictor) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/")
    def index():
        if WEBSITE.is_file():
            return send_from_directory(BASE_DIR, WEBSITE.name)
        return jsonify({"status": "PlantGuard AI API", "website": "not found"}), 404

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok" if predictor.ready else "model_not_loaded",
            "model_loaded": predictor.ready,
            "device": str(DEVICE),
            "error": predictor.error,
        })

    @app.get("/api/model")
    def model_info():
        return jsonify({
            "loaded": predictor.ready,
            "num_classes": len(predictor.class_names),
            "class_names": predictor.class_names,
            "device": str(DEVICE),
            **predictor.meta,
        })

    @app.get("/api/classes")
    def classes():
        return jsonify({
            "total_classes": len(predictor.class_names),
            "classes": [
                {"id": i, "key": name, **describe(name)}
                for i, name in enumerate(predictor.class_names)
            ],
        })

    @app.post("/api/predict")
    def predict():
        if not predictor.ready:
            return jsonify({"success": False, "error": predictor.error}), 503

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"success": False, "error": "Empty filename"}), 400

        if Path(file.filename).suffix.lower() not in ALLOWED_EXT:
            return jsonify({
                "success": False,
                "error": "Unsupported format. Use JPG, PNG, WEBP or BMP.",
            }), 400

        try:
            result = predictor.predict(file.read())
            return jsonify({"success": True, "result": result})
        except Exception as exc:  # noqa: BLE001
            log.exception("Prediction failed")
            return jsonify({"success": False, "error": str(exc)}), 500

    @app.get("/models/<path:filename>")
    def download_model(filename):
        return send_from_directory(MODEL_DIR, filename, as_attachment=True)

    return app


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlantGuard AI backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    predictor = Predictor(Path(args.model_path))
    app = create_app(predictor)

    print("\n" + "=" * 62)
    print("  PlantGuard AI - Plant Disease Detection")
    print("=" * 62)
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {'loaded [OK]' if predictor.ready else 'NOT LOADED [X]'}")
    if predictor.error:
        print(f"  Issue  : {predictor.error.splitlines()[0]}")
    print(f"  URL    : http://{args.host}:{args.port}")
    print("=" * 62 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
