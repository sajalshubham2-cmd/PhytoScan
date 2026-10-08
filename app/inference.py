import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

CLASS_NAMES = [
    "Tomato___Early_blight",                              # idx 0
    "Tomato___Late_blight",                               # idx 1
    "Tomato___Septoria_leaf_spot",                        # idx 2
    "Tomato___healthy",                                   # idx 3
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", # idx 4
    "Corn_(maize)___Common_rust_",                        # idx 5
    "Corn_(maize)___Northern_Leaf_Blight",                # idx 6
    "Corn_(maize)___healthy",                             # idx 7
]

CROP_INDICES = {
    "tomato": [0, 1, 2, 3],
    "maize":  [4, 5, 6, 7],
}

T_CROP_MASS = 0.35   # Minimum raw probability mass on selected crop
T_CONF      = 0.65   # Minimum renormalized top-1 probability
T_MARGIN    = 0.15   # Minimum margin (p_top1 - p_top2)


def preprocess_image(file_stream) -> np.ndarray:
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_leaf_disease(model: tf.keras.Model, image_batch: np.ndarray, selected_crop: str, kb: dict) -> dict:
    raw_probs = model.predict(image_batch, verbose=0)[0]
    valid_idxs = CROP_INDICES[selected_crop]

    raw_crop_mass = float(np.sum(raw_probs[valid_idxs]))
    if raw_crop_mass < T_CROP_MASS:
        other_crop = "Maize" if selected_crop == "tomato" else "Tomato"
        return {
            "status": "crop_mismatch",
            "crop_selected": selected_crop,
            "disease_key": None,
            "disease_name": "Crop Mismatch Detected",
            "confidence": round(raw_crop_mass, 4),
            "top2_margin": 0.0,
            "message": (
                f"The uploaded image does not appear to be a {selected_crop.capitalize()} leaf "
                f"(only {raw_crop_mass*100:.1f}% probability mass on {selected_crop}; likely {other_crop}). "
                "Please verify the selected crop or retake a clear close-up photo of a single leaf."
            ),
            "remedy": None,
        }

    crop_probs = raw_probs[valid_idxs] / raw_crop_mass
    sorted_local_order = np.argsort(crop_probs)[::-1]

    top1_local_idx = int(sorted_local_order[0])
    top2_local_idx = int(sorted_local_order[1])

    p1 = float(crop_probs[top1_local_idx])
    p2 = float(crop_probs[top2_local_idx])
    margin = p1 - p2

    top1_global_idx = valid_idxs[top1_local_idx]
    top2_global_idx = valid_idxs[top2_local_idx]
    top1_key = CLASS_NAMES[top1_global_idx]
    top2_key = CLASS_NAMES[top2_global_idx]

    if p1 < T_CONF or margin < T_MARGIN:
        return {
            "status": "low_confidence",
            "crop_selected": selected_crop,
            "disease_key": None,
            "disease_name": "No Disease Confidently Detected",
            "confidence": round(p1, 4),
            "top2_candidate": top2_key,
            "top2_confidence": round(p2, 4),
            "top2_margin": round(margin, 4),
            "message": (
                "No disease confidently detected. Please retake the photo in bright natural "
                "daylight with a single leaf centered and in sharp focus."
            ),
            "remedy": None,
        }

    entry = kb[top1_key]
    is_healthy = top1_key.endswith("healthy")
    status = "confident_healthy" if is_healthy else "confident_disease"
    msg = (
        f"Healthy {selected_crop.capitalize()} leaf detected ({p1*100:.1f}% confidence). No disease symptoms found."
        if is_healthy else
        f"Detected {entry['disease_name']} with {p1*100:.1f}% confidence. Review recommended remedies below."
    )

    return {
        "status": status,
        "crop_selected": selected_crop,
        "disease_key": top1_key,
        "disease_name": entry["disease_name"],
        "confidence": round(p1, 4),
        "top2_candidate": top2_key,
        "top2_confidence": round(p2, 4),
        "top2_margin": round(margin, 4),
        "message": msg,
        "remedy": entry,
    }
