import uvicorn

MODEL_PATH = "assets/model/skin_disease_model.tflite"

if __name__ == "__main__":
    uvicorn api:app --host 0.0.0.0 --port $PORT
