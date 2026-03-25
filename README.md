# 🤖 Pothole AI Detection System

AI-powered pothole detection and severity classification
using YOLOv8 and PyTorch.

## Features
- ✅ Validates if uploaded photo is a road photo
- ✅ Detects potholes in the image
- ✅ Classifies severity: HIGH / MEDIUM / LOW
- ✅ Returns confidence score
- ✅ FastAPI REST endpoint for integration

## Tech Stack
- YOLOv8 (Object Detection)
- PyTorch (Deep Learning)
- FastAPI (API Server)
- OpenCV (Image Processing)
- Python 3.10+

## Setup
```bash
pip install -r requirements.txt
```

## API Usage
```
POST /predict
Body: { image: file }
Response: {
  isRoad: true,
  severity: "HIGH",
  confidence: 96%
}
```