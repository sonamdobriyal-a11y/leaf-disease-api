from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "keras_model.h5"
LABELS_PATH = HERE / "labels.txt"


def _load_labels() -> List[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label file not found at {LABELS_PATH}")

    labels: List[str] = []
    with LABELS_PATH.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            line = line.strip()
            if not line:
                continue
            # Teachable Machine exports use `index label`, handle both cases.
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    if not labels:
        raise ValueError("Label file is empty.")

    return labels


def _load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model file not found at {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH))


app = FastAPI(title="Leaf Disease Classifier", version="1.0.0")

model = _load_model()
labels = _load_labels()

input_shape = model.input_shape
if len(input_shape) != 4:
    raise ValueError(
        f"Expected model input shape to be (batch, height, width, channels), got {input_shape}."
    )

_, input_height, input_width, input_channels = input_shape

if input_channels != 3:
    raise ValueError(f"Expected 3-channel RGB input; received {input_channels}.")


@app.get("/")
def read_root() -> JSONResponse:
    return JSONResponse(
        {
            "message": "Leaf Disease prediction service running.",
            "predict_endpoint": "/predict",
            "expected_image_size": [input_height, input_width],
            "labels": labels,
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to process the uploaded image.") from exc

    try:
        resample_method = Image.Resampling.BILINEAR  # Pillow >= 9
    except AttributeError:
        resample_method = Image.BILINEAR

    image = image.resize((input_width, input_height), resample_method)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0
    batched_input = np.expand_dims(image_array, axis=0)

    predictions = model.predict(batched_input)
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise RuntimeError(f"Unexpected prediction shape: {predictions.shape}")

    prediction_vector = predictions[0]
    predicted_index = int(np.argmax(prediction_vector))
    confidence = float(prediction_vector[predicted_index])
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else str(predicted_index)

    return JSONResponse(
        {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_confidences": {
                labels[idx] if idx < len(labels) else str(idx): float(score)
                for idx, score in enumerate(prediction_vector)
            },
        }
    )


@app.on_event("startup")
async def warmup_model() -> None:
    dummy_input = np.zeros((1, input_height, input_width, input_channels), dtype=np.float32)
    model.predict(dummy_input)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
