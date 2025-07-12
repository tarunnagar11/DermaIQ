# utils.py

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

def predict_burn(model, image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    pred_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return pred_class, confidence
