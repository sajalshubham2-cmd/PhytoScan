import os
import json
import time
from flask import Flask, request, jsonify
import tensorflow as tf
from app.inference import preprocess_image, predict_leaf_disease, CROP_INDICES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/crop_disease_mobilenetv2.keras")

with open(KB_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

MODEL = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "classes_count": len(KNOWLEDGE_BASE)
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    crop = (request.form.get("crop") or "").strip().lower()
    if crop not in CROP_INDICES:
        return jsonify({
            "status": "error",
            "error_code": "INVALID_CROP",
            "message": "Field 'crop' must be either 'tomato' or 'maize'."
        }), 400

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "error_code": "MISSING_IMAGE",
            "message": "No image file provided."
        }), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "error_code": "INVALID_FILE_TYPE",
            "message": "Unsupported file format. Please upload a .jpg, .jpeg, or .png leaf photo under 5 MB."
        }), 415

    t0 = time.perf_counter()
    batch = preprocess_image(file.stream)
    result = predict_leaf_disease(MODEL, batch, crop, KNOWLEDGE_BASE)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
