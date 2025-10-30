# Leaf Disease FastAPI Service

This project exposes the trained Keras model found in `keras_model.h5` through a FastAPI service that accepts image uploads and returns the predicted leaf disease label.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate               # On Windows PowerShell
pip install -r requirements.txt
```

> **Note**: Installing `tensorflow` can take a while and requires Python 3.9–3.11.
> With Python 3.9 you must pin `tensorflow==2.15.*`. TensorFlow 2.20+ needs Python 3.10 or newer.

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` to open the interactive Swagger UI where you can upload an image and trigger the `/predict` endpoint.

## Prediction Response

The `/predict` endpoint returns:
- `predicted_label`: Name of the most likely disease class.
- `confidence`: Confidence score (0–1) for the predicted label.
- `all_confidences`: Mapping of every label to its confidence score.

Ensure the uploaded images are clear leaf photos to get reliable predictions.
