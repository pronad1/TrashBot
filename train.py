from ultralytics import YOLO
import os

print("🚀 Starting YOLO training...")

# pretrained model
model = YOLO("yolov8n.pt")

# TRAIN (AUTO)
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

print("✅ Training done!")
print("Model saved in runs/detect/train/weights/best.pt")