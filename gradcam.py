import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize image for model prediction.
    """
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given input image and model.
    
    Args:
        img_array: Preprocessed image array with shape (1, H, W, C)
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Optional class index for which to compute heatmap

    Returns:
        heatmap: 2D normalized heatmap array
    """
    # Create a model that maps input -> conv layer -> predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    # Compute gradients of class output w.r.t. conv layer output
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap (prevent divide-by-zero)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap onto the original image.

    Args:
        image: Original PIL image
        heatmap: Normalized heatmap (2D array)
        alpha: Transparency of the heatmap overlay
        colormap: OpenCV colormap to apply (default: JET)

    Returns:
        RGB image with overlayed heatmap (NumPy array)
    """
    # Convert PIL to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Overlay heatmap onto image
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img
