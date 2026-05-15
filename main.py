import os
import requests
from pathlib import Path

MODEL_PATH = "/app/assets/model/skin_disease_model.tflite"
GDRIVE_FILE_ID = "1qr50bnMKsua4NhxyER_lmzXL8f5t_Ped"

def download_model():
    if Path(MODEL_PATH).exists():
        return
    print("⬇️ Downloading model...")
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"
    session = requests.Session()
    r = session.get(url, stream=True)
    # Handle Google's virus scan warning page for large files
    for key, value in r.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={value}"
            r = session.get(url, stream=True)
            break
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded!")

download_model()  # runs once on startup

# ADD THIS LINE - Import and expose the FastAPI app
from api import app
