# Image Classifier App

A small full-stack image classifier with a Flask API and a React frontend. Upload a JPG, PNG, or WebP image and receive the model's top predictions with confidence scores.

## Project structure

```
backend/   Flask API and ImageNet classifier
frontend/  React + Vite user interface
```

## Quick start

### 1. Start the API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

The API runs at `http://127.0.0.1:5000`.

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (normally `http://127.0.0.1:5173`). The dev server forwards `/api` requests to Flask.

## API

`POST /api/classify` accepts a multipart form upload with an `image` field and returns:

```json
{
  "predictions": [
    { "label": "golden retriever", "confidence": 92.34 }
  ]
}
```

The first request downloads the ImageNet-pretrained MobileNetV3 weights if they are not already cached. To run the server without PyTorch or model downloads, set `CLASSIFIER_MODE=demo`; it will return a clearly marked visual-analysis demo result instead.

## Checks

```bash
cd backend && python -m unittest discover -s tests
cd frontend && npm run build
```
