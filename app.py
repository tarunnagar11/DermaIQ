import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import preprocess_image, predict_burn

# For MobileNetV2
def process_mobilenet(image):
    from gradcam import (
        preprocess_image as gradcam_preprocess_mobilenet,
        make_gradcam_heatmap as make_gradcam_heatmap_mobilenet,
        overlay_heatmap as overlay_heatmap_mobilenet
    )
    model = tf.keras.models.load_model("burn_model_final.h5", compile=False)
    img_array = gradcam_preprocess_mobilenet(image)
    last_conv_layer = "block_16_project"
    heatmap = make_gradcam_heatmap_mobilenet(img_array, model, last_conv_layer)
    gradcam_img = overlay_heatmap_mobilenet(image, heatmap)
    return model, gradcam_img

# For Custom CNN
def process_custom_cnn(image):
    from gradcamcustomcnn import (
        preprocess_image as gradcam_preprocess_custom,
        make_gradcam_heatmap as make_gradcam_heatmap_custom,
        overlay_heatmap as overlay_heatmap_custom
    )
    model = tf.keras.models.load_model("burn_model_customcnn_functional.h5", compile=False)
    model.build(input_shape=(None, 224, 224, 3))
    _ = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
    img_array = gradcam_preprocess_custom(image, (224, 224))
    last_conv_layer = "conv2d_5"
    heatmap = make_gradcam_heatmap_custom(img_array, model, last_conv_layer)
    original_img = np.array(image.resize((224, 224)))
    gradcam_img = overlay_heatmap_custom(heatmap, original_img)
    return model, gradcam_img

from model import init_chat
import os

# ---------------- Streamlit Styling ----------------
st.set_page_config(page_title="DermaIQ - Burn Classifier & Assistant", layout="centered")

# ---------------- Sidebar ----------------
st.sidebar.title("🔥 DermaIQ - Burn Classifier & Assistant")

st.sidebar.info("""
AI-powered burn severity detection & medical chatbot.

**Model:** MobileNetV2 , Custom CNN , Gemini Chatbot  
**Team:** DermaIQ  
""")

selected_option = st.sidebar.radio(
    "Choose an option:",
    ("🔥 Burn Classifier", "📈 Model Evaluation", "💬 Burn & Medical Chatbot")
)

# ---------------- Burn Classifier ----------------
if selected_option == "🔥 Burn Classifier":
    st.title("🩺 Burn Severity Classifier")

    model_choice = st.radio(
        "Choose Model",
        ("MobileNetV2", "Custom CNN"),
        horizontal=True
    )

    uploaded_file = st.file_uploader("Upload a wound image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            if model_choice == "MobileNetV2":
                model, gradcam_img = process_mobilenet(image)
            else:
                model, gradcam_img = process_custom_cnn(image)

            pred_class, confidence = predict_burn(model, image)

            burn_degrees = ["First-degree Burn", "Second-degree Burn", "Third-degree Burn"]
            recommendations = [
                "Apply cool water or aloe vera. Seek medical advice if needed.",
                "Cover with clean dressing. Medical attention recommended.",
                "Severe burn. Seek **immediate** medical attention."
            ]

            with st.expander("🧬 Prediction & Recommendation", expanded=True):
                st.success(f"Prediction: **{burn_degrees[pred_class]}**")
                st.progress(confidence)
                st.markdown(f"<div style='text-align:center; margin-top:-12px; font-size:14px;'>Model Confidence: <b>{confidence * 100:.2f}%</b></div>", unsafe_allow_html=True)
                st.info(f"**Recommendation:** {recommendations[pred_class]}")

            with st.expander("🔬 Grad-CAM Heatmap"):
                st.image(gradcam_img, caption="Model Focus Area (Grad-CAM)", use_column_width=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    else:
        st.warning("Please upload a wound image to get started.")

# ---------------- Model Evaluation Page ----------------
elif selected_option == "📈 Model Evaluation":
    st.title("📈 Model Evaluation Dashboard")

    eval_option = st.radio(
        "Select Evaluation Metric:",
        ("📊 Accuracy & Loss Graph", "🧹 Confusion Matrix"),
        horizontal=True
    )

    if eval_option == "📊 Accuracy & Loss Graph":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MobileNetV2")
            if os.path.exists("training_metrics.png"):
                st.image("training_metrics.png", caption="MobileNetV2 Accuracy & Loss")
            else:
                st.warning("MobileNetV2 metrics not found.")

        with col2:
            st.subheader("Custom CNN")
            if os.path.exists("training_metrics_custom_cnn.png"):
                st.image("training_metrics_custom_cnn.png", caption="Custom CNN Accuracy & Loss")
            else:
                st.warning("Custom CNN metrics not found.")

    elif eval_option == "🧹 Confusion Matrix":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MobileNetV2")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="MobileNetV2 Confusion Matrix")
            else:
                st.warning("MobileNetV2 confusion matrix not found.")

        with col2:
            st.subheader("Custom CNN")
            if os.path.exists("confusion_matrix_custom_cnn.png"):
                st.image("confusion_matrix_custom_cnn.png", caption="Custom CNN Confusion Matrix")
            else:
                st.warning("Custom CNN confusion matrix not found.")

# ---------------- Integrated Chatbot ----------------
elif selected_option == "💬 Burn & Medical Chatbot":
    st.title("💬 DermaIQ - Burn & Medical Assistant")

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = init_chat()
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask your burn or medical-related question:")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Generating response..."):
            response = st.session_state.chat_session.send_message(user_input)
            bot_response = response.text

        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
