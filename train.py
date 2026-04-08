from ultralytics import YOLO
import torch
import os

# Check if GPU available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Using device: {device}")

if device == 'cpu':
    print("⚠️  No GPU found - training on CPU (will be slow)")
    print("💡 Tip: Use Google Colab for faster training!")
else:
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

# Load YOLOv8 model
# yolov8n = nano (fastest, less accurate)
# yolov8s = small (good balance) ← we use this
# yolov8m = medium (more accurate, slower)
# yolov8l = large (most accurate, slowest)
model = YOLO('yolov8s.pt')

print("\n🚀 Starting training...")
print("━" * 50)

# Train the model
results = model.train(
    data=os.path.abspath('dataset_yolo/data.yaml'),
    epochs=50,           # number of training rounds
    imgsz=640,           # image size
    batch=16,            # images per batch
    patience=10,         # stop if no improvement
    device=device,
    project='runs/train',
    name='pothole_severity',
    exist_ok=True,
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    plots=True,          # save training plots
    save=True,           # save best model
    verbose=True
)

print("\n✅ Training complete!")
print(f"📁 Results saved to: runs/train/pothole_severity/")
print(f"📊 Best model: runs/train/pothole_severity/weights/best.pt")