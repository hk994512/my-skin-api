import os, requests
from pathlib import Path

MODEL_PATH = "assets/models/skin_disease_model.tflite"
GDRIVE_FILE_ID = "1qr50bnMKsua4NhxyER_lmzXL8f5t_Ped"  # from the sharing link

def download_model():
    if Path(MODEL_PATH).exists():
        return
    print("⬇️ Downloading model...")
    Path("assets/models").mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded!")

download_model()  # runs once on startup