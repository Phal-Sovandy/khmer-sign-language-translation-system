# Backend - Khmer Sign Language Translation System

<div align="center">

![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.8-4285F4?logo=google)

**Flask backend server providing ML-powered sign language gesture recognition API**

[API Documentation](#api-endpoints) • [Setup](#installation) • [Usage](#usage) • [Model Information](#model-architecture)

</div>

---

## Overview

The backend server is a Flask-based REST API that processes sign language gestures in real-time. It uses MediaPipe for hand landmark detection and a PyTorch-based deep learning model (LandmarkNet) to classify Khmer sign language gestures from hand landmark data.

### Key Features

- **Real-time Processing**: Fast inference using PyTorch with GPU support (CUDA)
- **Hand Landmark Detection**: MediaPipe integration for accurate hand tracking
- **RESTful API**: Simple JSON-based API for frontend integration
- **CORS Enabled**: Configured for cross-origin requests from frontend
- **Model Inference**: Pre-trained LandmarkNet model for gesture classification

## Tech Stack

- **Flask 3.0.0** - Python web framework
- **PyTorch 2.1.0** - Deep learning framework for model inference
- **MediaPipe 0.10.8** - Hand landmark detection
- **OpenCV 4.8.1.78** - Image processing and manipulation
- **NumPy 1.24.3** - Numerical computing
- **Flask-CORS 4.0.0** - Cross-origin resource sharing

## Installation

### Prerequisites

- **Python 3.11** or higher
- **pip** (Python package manager)
- **CUDA** (optional, for GPU acceleration)

### Setup Instructions

1. **Navigate to the backend directory**

   ```bash
   cd backend
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

   **Activate the virtual environment:**

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you have CUDA installed and want GPU acceleration, you may need to install PyTorch with CUDA support separately:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify the model file**

   Ensure that the trained model file (`model_0.pth` or similar) exists in the `model/` directory. The server will automatically load the model on startup.

## Usage

### Starting the Server

1. **Activate the virtual environment** (if not already activated)

   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Run the server**

   ```bash
   python server.py
   ```

   The server will start on `http://localhost:3000`

   You should see output like:

   ```
   ✅ LandmarkNet loaded
   ✅ Starting Flask server...
   * Running on http://0.0.0.0:3000
   ```

### Server Configuration

The server runs with the following default settings:

- **Host**: `0.0.0.0` (accessible from all network interfaces)
- **Port**: `3000`
- **Debug Mode**: Enabled (for development)

To change these settings, modify the last line in `server.py`:

```python
app.run(host='0.0.0.0', port=3000, debug=True)
```

### CORS Configuration

The server is configured to accept requests from the following origins:

- `http://localhost:5173` (Vite dev server)
- `http://127.0.0.1:5173`
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:5000`
- `http://127.0.0.1:5000`
- `http://localhost:5500`
- `http://127.0.0.1:5500`

To add additional origins, modify the CORS configuration in `server.py`.

## API Endpoints

### GET `/`

Serves the index page (HTML template).

**Response:** HTML page

---

### POST `/predict_image`

Predicts sign language gesture from an image containing hand landmarks.

**Request Headers:**

```
Content-Type: application/json
```

**Request Body:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

The `image` field should contain a base64-encoded image with data URI prefix (e.g., `data:image/jpeg;base64,`).

**Response (Success):**

```json
{
  "label": "hello",
  "confidence": 95.5,
  "landmarks": [
    {"x": 0.123, "y": 0.456},
    {"x": 0.234, "y": 0.567},
    ...
  ]
}
```

**Response Fields:**

- `label` (string): The predicted sign language gesture name
- `confidence` (float): Prediction confidence percentage (0-100)
- `landmarks` (array): Normalized hand landmark coordinates (x, y) from MediaPipe

**Response (Error):**

```json
{
  "error": "No hand detected"
}
```

**Example using cURL:**

```bash
curl -X POST http://localhost:3000/predict_image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

**Example using Python:**

```python
import requests
import base64

# Read and encode image
with open("hand_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Send request
response = requests.post(
    "http://localhost:3000/predict_image",
    json={"image": f"data:image/jpeg;base64,{image_data}"}
)

result = response.json()
print(f"Predicted: {result['label']} ({result['confidence']:.2f}%)")
```

## Model Architecture

### LandmarkNet

The backend uses a custom PyTorch neural network called `LandmarkNet` for gesture classification.

**Architecture:**

```
Input: 42 features (21 landmarks × 2 coordinates: x, y)
  ↓
Linear(42 → 128) + ReLU + Dropout(0.3)
  ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
  ↓
Linear(64 → num_classes)
  ↓
Output: Class probabilities
```

**Key Features:**

- Input: 42 normalized landmark coordinates (21 hand landmarks × 2 coordinates)
- Hidden layers: 128 and 64 neurons with ReLU activation
- Dropout regularization: 0.3 and 0.2 to prevent overfitting
- Output: Probability distribution over sign language classes

### Model Loading

The model is automatically loaded from `model/model_0.pth` on server startup. The checkpoint file contains:

- `model_state_dict`: Trained model weights
- `class_names`: List of sign language gesture names

### Hand Landmark Processing

1. **Detection**: MediaPipe detects hand landmarks in the input image
2. **Normalization**: Landmark coordinates are normalized (mean-centered and scaled)
3. **Feature Extraction**: 42 features are extracted (21 landmarks × 2 coordinates)
4. **Prediction**: LandmarkNet processes the features and outputs class probabilities
5. **Result**: Top prediction with confidence score is returned

## Project Structure

```
backend/
├── model/                    # Trained model files
│   └── model_0.pth          # PyTorch model checkpoint
├── templates/                # HTML templates
│   └── index.html           # Index page template
├── server.py                 # Main Flask application
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Development

### Running in Development Mode

The server runs in debug mode by default, which provides:

- Automatic reloading on code changes
- Detailed error messages
- Interactive debugger (if errors occur)

### Testing the API

You can test the API using:

1. **cURL** (command line)
2. **Postman** (GUI tool)
3. **Python requests** library
4. **Frontend application** (React app)

### Debugging

- Check server logs for error messages
- Verify model file exists in `model/` directory
- Ensure all dependencies are installed correctly
- Check CORS configuration if experiencing frontend connection issues

## Performance

### Optimization Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster inference
2. **Batch Processing**: For multiple predictions, consider batching requests
3. **Model Quantization**: Use PyTorch quantization for smaller model size
4. **Caching**: Implement caching for frequently requested predictions

### Current Performance

- **Inference Time**: ~10-50ms per image (CPU)
- **Inference Time**: ~5-20ms per image (GPU with CUDA)
- **Model Size**: Varies based on number of classes

## Troubleshooting

### Common Issues

**Issue: Model not loading**

- **Solution**: Ensure `model_0.pth` exists in the `model/` directory
- Check file permissions

**Issue: Import errors**

- **Solution**: Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**Issue: CORS errors from frontend**

- **Solution**: Check that frontend origin is in the CORS allowed origins list
- Verify server is running and accessible

**Issue: No hand detected**

- **Solution**: Ensure image contains visible hand landmarks
- Check image quality and lighting conditions
- Verify MediaPipe is detecting hands correctly

**Issue: CUDA/GPU not working**

- **Solution**: Install CUDA-enabled PyTorch version
- Verify CUDA drivers are installed
- Check `torch.cuda.is_available()` returns `True`

## Deployment

### Production Considerations

1. **Disable Debug Mode**: Set `debug=False` in production
2. **Use Production WSGI Server**: Consider using Gunicorn or uWSGI
3. **Environment Variables**: Use environment variables for configuration
4. **HTTPS**: Enable HTTPS for secure connections
5. **Rate Limiting**: Implement rate limiting to prevent abuse
6. **Error Logging**: Set up proper logging and monitoring

### Example with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 server:app
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "server.py"]
```

## Contributing

For contribution guidelines, see the main [CONTRIBUTING.md](../CONTRIBUTING.md) file.

### Backend-Specific Contributions

- Model improvements and optimizations
- API endpoint enhancements
- Performance optimizations
- Error handling improvements
- Documentation updates

## License

This project is licensed under the ISC License - see the [LICENSE.md](../LICENSE.md) file for details.

## Related Documentation

- [Main README](../README.md) - Project overview and frontend documentation
- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Complete project structure

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [Report a bug or request a feature](https://github.com/Phal-Sovandy/Khmer-Sign-Language-Translation-System/issues)
- **Email**: phalsovandy007@gmail.com

---

<div align="center">

**Backend API for Khmer Sign Language Translation System**

Part of the [Khmer Sign Language Translation System](../README.md) project

</div>
