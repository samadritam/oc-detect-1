import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware 

model = tf.keras.models.load_model("oral_cancer_cnn.h5")

class_labels = ["CANCER", "NORMAL"]

app = FastAPI(title="Oral Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    if 0.3 <= confidence <= 0.7:
        label = "DOUBTFUL"
    else:
        label = class_labels[predicted_class]

    return label, confidence

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    label, confidence = predict(img)

    return {
        "image_name": file.filename,
        "label": label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

