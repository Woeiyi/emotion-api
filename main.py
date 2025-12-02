from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from utils import preprocess_frame
import uvicorn

# ================================
# CONFIG
# ================================
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 224

# Positive / Negative Mapping
POSITIVE = {"happy", "surprise"}
NEGATIVE = {"angry", "disgust", "fear", "sad"}
NEUTRAL = {"neutral"}

# ================================
# LOAD TFLITE MODEL
# ================================
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================================
# FASTAPI SETUP
# ================================
app = FastAPI()

# ================================
# REQUEST MODEL
# ================================
class FrameRequest(BaseModel):
    frame_base64: str
    frame_index: int


# ================================
# ROUTES
# ================================
@app.get("/")
def index():
    return {"status": "Emotion API is running."}


@app.post("/predict-frame")
def predict_frame(data: FrameRequest):
    # 1. Decode base64 → PIL image
    try:
        img_bytes = base64.b64decode(data.frame_base64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except:
        return {"error": "Invalid base64 image"}

    # 2. Preprocess frame
    input_tensor = preprocess_frame(img)

    # 3. Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0].astype(float)

    # 4. Get raw predicted emotion (from 7 classes)
    raw_pred = LABELS[int(np.argmax(output))]

    # ================================
    # 5. Convert to POSITIVE / NEGATIVE / NEUTRAL
    # ================================
    if raw_pred in POSITIVE:
        sentiment = "positive"
    elif raw_pred in NEGATIVE:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # 6. Build response
    response = {
        "frame_index": data.frame_index,
        "predicted_class": sentiment,  # <= simplified
        "raw_emotion": raw_pred,       # <= still included (optional)
        "confidences": {
            LABELS[i]: float(output[i]) for i in range(len(LABELS))
        }
    }

    return response


# ================================
# FOR LOCAL TESTING
# ================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
