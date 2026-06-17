import os

print("🚀 Downloading TACO dataset...")

os.system("git clone https://github.com/pedropro/TACO.git")

print("🚀 Installing YOLO environment done")

print("🚀 Creating dataset folders...")
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

print("✅ Setup complete")