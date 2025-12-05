# Backend - Sign Language Translation System

This is the Flask backend server for the Sign Language Translation System.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python server.py
```

The server will start on `http://localhost:3000`

## API Endpoints

- `GET /` - Serves the index page
- `POST /predict_image` - Predicts sign language from an image
  - Request body: `{ "image": "data:image/jpeg;base64,..." }`
  - Response: `{ "label": "sign_name", "confidence": 95.5, "landmarks": [...] }`

## Model

The trained model files are stored in the `model/` directory.
