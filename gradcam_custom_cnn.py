# gradcam_custom_cnn.py - This looks correct and updated!
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize the input image.
    """
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a Sequential or Functional CNN model.
    """
    # Ensure the model is built before proceeding.
    # This is crucial for Sequential models loaded with compile=False
    # and for accessing layer outputs by name.
    if not model.built:
        model.build(input_shape=img_array.shape) # Or (None, 224, 224, 3) <-- THIS IS THE KEY FIX

    # Step 1: Auto-detect last Conv2D layer if name not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in the model.")

    # Step 2: Create Grad-CAM model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Step 3: Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    heatmap = heatmap / max_val if max_val != 0 else np.zeros_like(heatmap)

    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the Grad-CAM heatmap onto the input image.
    """
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img)