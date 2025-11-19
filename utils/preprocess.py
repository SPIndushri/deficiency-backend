from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
