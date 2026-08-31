"""Flask API for classifying image uploads with ImageNet MobileNetV3."""

from __future__ import annotations

import io
import os
from functools import lru_cache

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename: str) -> bool:
    """Return whether filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@lru_cache(maxsize=1)
def get_model():
    """Load MobileNet only once, avoiding startup work until first upload."""
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model, weights, torch


def demo_prediction(image: Image.Image) -> list[dict[str, float | str]]:
    """Provide a usable offline fallback when explicitly enabled for demos."""
    rgb = image.convert("RGB").resize((1, 1))
    red, green, blue = rgb.getpixel((0, 0))
    labels = [(red, "warm-toned image"), (green, "natural / green scene"), (blue, "cool-toned image")]
    labels.sort(reverse=True)
    return [
        {"label": f"{label} (demo analysis)", "confidence": round(58 - position * 13 + value / 255 * 20, 2)}
        for position, (value, label) in enumerate(labels)
    ]


def classify(image: Image.Image) -> list[dict[str, float | str]]:
    """Return the three most likely ImageNet labels for an opened image."""
    if os.getenv("CLASSIFIER_MODE", "").lower() == "demo":
        return demo_prediction(image)

    model, weights, torch = get_model()
    tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.nn.functional.softmax(model(tensor)[0], dim=0)
    top_probabilities, top_categories = torch.topk(probabilities, 3)
    return [
        {
            "label": weights.meta["categories"][category.item()].replace("_", " "),
            "confidence": round(probability.item() * 100, 2),
        }
        for probability, category in zip(top_probabilities, top_categories)
    ]


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/classify")
def classify_image():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image to classify."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Use a JPG, PNG, or WebP image."}), 400

    try:
        contents = uploaded.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "That file could not be read as an image."}), 400

    try:
        predictions = classify(image)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "That file could not be read as an image."}), 400
    except Exception as exc:  # Model downloads and runtime errors should be actionable to users.
        app.logger.exception("Classification failed")
        return jsonify({"error": f"Classification could not run: {exc}"}), 503

    return jsonify({"predictions": predictions})


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image must be 8 MB or smaller."}), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
