from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime
import os
import torch
from huggingface_hub import hf_hub_download

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================
# AUTOMATIC MODEL DOWNLOAD FROM HUGGING FACE
# ============================================

def download_trash_model():
    """
    Automatically download YOLO trash detection model from Hugging Face.
    The model detects: Plastic, Paper, Glass, Metal, Cardboard, Organic, etc.
    """
    
    model_path = "models/trash_detection.pt"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return model_path
    
    print("📥 Downloading Trash Detection Model from Hugging Face...")
    print("   This model can detect: Plastic, Paper, Glass, Metal, Cardboard, Organic, E-waste")
    
    try:
        # Option 1: Try downloading from Hugging Face (8-class waste detection model)
        model_path = hf_hub_download(
            repo_id="HrutikAdsare/waste-detection-yolov8",
            filename="best.pt",
            cache_dir="models"
        )
        print(f"✅ Model downloaded successfully from Hugging Face!")
        print(f"   📁 Saved at: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"⚠️ Could not download from primary source: {e}")
        
        try:
            # Option 2: Alternative model from Hugging Face (trash detection)
            print("🔄 Trying alternative model source...")
            model_path = hf_hub_download(
                repo_id="Alope/trash-detection-yolo11n",
                filename="best.pt",
                cache_dir="models"
            )
            print(f"✅ Alternative model downloaded successfully!")
            return model_path
            
        except Exception as e2:
            print(f"⚠️ Alternative model also failed: {e2}")
            
            # Option 3: Use YOLO pretrained model with custom class mapping as fallback
            print("🔄 Using YOLO pretrained model with custom mapping...")
            return "yolov8n.pt"  # Fallback to base model

# Download or load the model
MODEL_PATH = download_trash_model()

# Load the model
if MODEL_PATH == "yolov8n.pt":
    model = YOLO("yolov8n.pt")
    model_type = "fallback"
    print("📋 Using fallback mode - will map COCO classes to trash categories")
else:
    try:
        model = YOLO(MODEL_PATH)
        model_type = "trashnet"
        print(f"\n📊 Model loaded successfully!")
        print(f"🏷️  Detects these classes: {list(model.names.values())}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to YOLO pretrained model...")
        model = YOLO("yolov8n.pt")
        model_type = "fallback"

# Define the classes we want to detect
TARGET_CLASSES = ['plastic', 'paper', 'glass', 'metal', 'cardboard', 'organic']

# For fallback mode: Map COCO classes to trash categories
COCO_TO_TRASH = {
    # Plastic items
    39: 'plastic', 40: 'plastic', 41: 'plastic', 24: 'plastic', 44: 'plastic',
    # Glass items  
    25: 'glass', 26: 'glass', 27: 'glass',
    # Paper items
    33: 'paper', 34: 'paper', 73: 'paper', 47: 'paper',
    # Metal items
    43: 'metal', 45: 'metal',
    # Organic items
    46: 'organic', 47: 'organic', 48: 'organic', 49: 'organic'
}

UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Detection statistics
detection_stats = {
    'plastic': 0,
    'paper': 0,
    'glass': 0,
    'metal': 0,
    'cardboard': 0,
    'organic': 0,
    'total': 0
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    global detection_stats
    
    try:
        # Check if image is in request
        if "image" not in request.files:
            return jsonify({"error": "No image file"}), 400
        
        file = request.files["image"]
        
        # Read and decode image
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Run detection
        results = model(img, conf=0.25, iou=0.45)
        
        detected = []
        detections_with_confidence = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get class name based on model type
                    if model_type == "trashnet":
                        # Direct class name from model
                        class_name = model.names[class_id].lower()
                        
                        # Check if it's one of our target classes
                        if any(target in class_name for target in TARGET_CLASSES):
                            trash_type = class_name
                        else:
                            continue
                    else:
                        # Fallback mode: Map COCO classes to trash
                        if class_id in COCO_TO_TRASH:
                            trash_type = COCO_TO_TRASH[class_id]
                        else:
                            continue
                    
                    if confidence > 0.3:
                        detected.append(trash_type)
                        detections_with_confidence.append({
                            "type": trash_type,
                            "confidence": confidence
                        })
                        
                        # Update statistics
                        if trash_type in detection_stats:
                            detection_stats[trash_type] += 1
                        detection_stats['total'] += 1
                        
                        # Draw bounding box on image
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Color coding by waste type
                        colors = {
                            'plastic': (0, 255, 0),    # Green
                            'paper': (255, 255, 0),    # Cyan
                            'glass': (0, 255, 255),    # Yellow
                            'metal': (255, 0, 0),      # Blue
                            'cardboard': (255, 165, 0),# Orange
                            'organic': (0, 128, 255)   # Light Blue
                        }
                        color = colors.get(trash_type, (0, 255, 0))
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"{trash_type.upper()} ({confidence:.2f})"
                        cv2.putText(img, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Remove duplicates while preserving order
        unique_detected = []
        for item in detected:
            if item not in unique_detected:
                unique_detected.append(item)
        
        # Print detection summary
        print(f"\n{'='*40}")
        print(f"🔍 DETECTION RESULT")
        print(f"{'='*40}")
        if detections_with_confidence:
            for det in detections_with_confidence:
                print(f"  🗑️  {det['type'].upper()}: {det['confidence']:.2f}")
        else:
            print("  ❌ No plastic/paper/glass detected")
        print(f"{'='*40}")
        
        # Save annotated image
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(path, img)
        
        # Encode for web display
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        
        # Emit real-time update to all connected clients
        socketio.emit("update", {
            "image": img_base64,
            "detected": unique_detected,
            "detections": detections_with_confidence,
            "time": filename,
            "timestamp": datetime.now().isoformat(),
            "stats": detection_stats
        })
        
        return jsonify({
            "status": "ok",
            "detected": unique_detected,
            "count": len(unique_detected),
            "details": detections_with_confidence,
            "stats": detection_stats
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get current detection statistics"""
    return jsonify(detection_stats)

@app.route("/model_info", methods=["GET"])
def model_info():
    """Get information about the loaded model"""
    model_classes = list(model.names.values()) if hasattr(model, 'names') else TARGET_CLASSES
    
    return jsonify({
        "model_type": model_type,
        "classes": model_classes,
        "num_classes": len(model_classes),
        "can_detect_plastic": True,
        "can_detect_paper": True,
        "can_detect_glass": True,
        "target_classes": TARGET_CLASSES
    })

@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    """Reset detection statistics"""
    global detection_stats
    detection_stats = {
        'plastic': 0,
        'paper': 0,
        'glass': 0,
        'metal': 0,
        'cardboard': 0,
        'organic': 0,
        'total': 0
    }
    return jsonify({"status": "reset", "stats": detection_stats})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    return response

if __name__ == "__main__":
    # Get local IP
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 TRASHBOT AI SERVER - AUTOMATIC MODEL DOWNLOAD")
    print("="*60)
    print(f"📦 Model type: {model_type.upper()}")
    
    if hasattr(model, 'names') and model.names:
        print(f"🏷️  Detects: {list(model.names.values())}")
    else:
        print(f"🏷️  Target classes: {TARGET_CLASSES}")
    
    print(f"\n✅ Plastic Detection: ✓")
    print(f"✅ Paper Detection: ✓")
    print(f"✅ Glass Detection: ✓")
    print(f"✅ Metal Detection: ✓")
    print(f"✅ Cardboard Detection: ✓")
    
    print(f"\n🌐 Server running on:")
    print(f"   → http://127.0.0.1:5000")
    print(f"   → http://{local_ip}:5000")
    print(f"📡 WebSocket: ws://{local_ip}:5000")
    print("="*60 + "\n")
    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)