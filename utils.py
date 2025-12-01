import numpy as np
from PIL import Image

# ================================
# PREPROCESS FRAME
# ================================
def preprocess_frame(image: Image.Image):
    """
    Takes a PIL image and returns a preprocessed tensor
    shaped (1, 224, 224, 3) normalized for EfficientNet.
    """

    # Resize to 224x224
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image).astype("float32")

    # Normalize EXACTLY like EfficientNet preprocess_input
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
