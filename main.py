from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.preprocess import preprocess_image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import os

# ----------------------------
# Load Model & Labels
# ----------------------------
MODEL_PATH = "model/FinalMP_model.keras"
LABELS_PATH = "model/label_classes.npy"

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).item()

# Convert dict → ordered list
label_list = [cls for cls, idx in sorted(labels.items(), key=lambda x: x[1])]

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Deficiency Detection API")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def home():
    return {"message": "API is live. Upload a nail image at /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file)

    # Preprocess
    arr = preprocess_image(image)

    # Model prediction
    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    # Prepare response
    result = {
        "predicted_class": label_list[top_idx],
        "confidence": round(confidence, 2),
        "raw_predictions": preds.tolist(),
    }

    return result

# ----------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
