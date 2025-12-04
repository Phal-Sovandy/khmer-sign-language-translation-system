from flask import Flask, render_template, Response, request
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from pathlib import Path
import base64
from flask_cors import CORS

# -------------------------
# Flask app + CORS
# -------------------------
app = Flask(__name__)
# Allow both localhost and 127.0.0.1 origins (your HTML test page)
from flask_cors import CORS

CORS(app, resources={r"/*": {
    "origins": [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]
}})


# -------------------------
# Config
# -------------------------
MODEL_PATH = Path(__file__).parent / "model" / "model_0.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LandmarkNet model
# -------------------------
class LandmarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------
# Load model
# -------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = LandmarkNet(num_classes).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ LandmarkNet loaded")

# -------------------------
# Mediapipe Hands
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------
# Prediction functions
# -------------------------
def extract_and_normalize_landmarks(hand_landmarks):
    xs = np.array([lm.x for lm in hand_landmarks.landmark])
    ys = np.array([lm.y for lm in hand_landmarks.landmark])
    xs = (xs - xs.mean()) / (xs.std() + 1e-6)
    ys = (ys - ys.mean()) / (ys.std() + 1e-6)
    features = np.concatenate([xs, ys]).astype(np.float32)
    return torch.tensor(features).unsqueeze(0).to(DEVICE)

def predict_hand(hand_landmarks):
    features = extract_and_normalize_landmarks(hand_landmarks)
    with torch.no_grad():
        output = model(features)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        return class_names[top_idx], probs[top_idx].item() * 100

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Convert to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run MediaPipe hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        label, conf = predict_hand(hand_landmarks)

        # Collect normalized landmark coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y})

        return {
            "label": label,
            "confidence": conf,
            "landmarks": landmarks
        }
    else:
        return {"error": "No hand detected"}


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(host='0.0.0.0', port=3000, debug=True)
