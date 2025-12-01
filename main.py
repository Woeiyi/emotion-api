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
    frame_base64: str  # Base64 string of frame
    frame_index: int   # Which frame number (optional but useful)


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

    # 2. Preprocess (resize, normalize, expand dims)
    input_tensor = preprocess_frame(img)

    # 3. Run inference through TFLite
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # 4. Convert to normal list
    output = output[0]
    output = output.astype(float)

    # 5. Build response
    response = {
        "frame_index": data.frame_index,
        "predicted_class": LABELS[int(np.argmax(output))],
        "confidences": {
            LABELS[i]: float(output[i]) for i in range(len(LABELS))
        }
    }

    return response


# ================================
# FOR LOCAL TESTING (Render ignores this)
# ================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
