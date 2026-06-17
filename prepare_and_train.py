import os
import pandas as pd
import requests
from tqdm import tqdm
from ultralytics import YOLO
import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE, "dataset/images/all_image_urls.csv")

# =========================
# AUTO OUTPUT FOLDER (NEW)
# =========================
RUN_DIR = os.path.join(BASE, "runs_auto")
IMG_DIR = os.path.join(BASE, "dataset/images/train")
VAL_DIR = os.path.join(BASE, "dataset/images/val")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

print("🚀 Loading CSV...")
df = pd.read_csv(CSV_PATH)

url_column = df.columns[0]

def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except:
        return False
    return False

print("🚀 Downloading images...")

count = 0
max_images = 500

for i, url in tqdm(enumerate(df[url_column])):
    if count >= max_images:
        break

    file_path = os.path.join(IMG_DIR, f"img_{i}.jpg")

    if download_image(url, file_path):
        count += 1

print(f"✅ Download complete: {count} images")

# =========================
# AUTO data.yaml
# =========================
yaml_path = os.path.join(BASE, "data.yaml")

yaml_content = f"""
path: {BASE}/dataset
train: images/train
val: images/train

nc: 4
names: ["bottle", "cup", "can", "other"]
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("🚀 Starting training...")

# =========================
# AUTO RUN NAME (VERY IMPORTANT)
# =========================
run_name = "trashbot_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model = YOLO("yolov8n.pt")

model.train(
    data=yaml_path,
    epochs=20,
    imgsz=640,
    batch=8,
    project=RUN_DIR,   # 👉 ALL OUTPUT HERE
    name=run_name,     # 👉 UNIQUE RUN FOLDER
    save=True,
    save_period=1
)

print("🎯 TRAINING DONE!")

print(f"""
📦 OUTPUT SAVED HERE:

{RUN_DIR}\\{run_name}

✔ best model
✔ last model
✔ logs
✔ graphs
✔ weights
""")