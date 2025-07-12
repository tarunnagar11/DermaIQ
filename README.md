# 🔥 DermaIQ - Burn Classifier & Medical Assistant

**DermaIQ** is an AI-powered web application that classifies skin burn severity using deep learning (MobileNetV2 or a Custom CNN), provides Grad-CAM heatmaps to visualize model predictions, and includes a Gemini-powered medical chatbot for real-time support and care advice.

---

## 🧠 Features

- 🔍 Classifies burn severity (First, Second, Third-degree)
- 🧠 Choose between **MobileNetV2** and **Custom CNN**
- 🌈 Grad-CAM heatmap visualization for model interpretability
- 📊 Evaluation dashboard (accuracy/loss graphs and confusion matrix)
- 💬 Gemini-powered chatbot for medical and burn-related queries

---

## 🗂️ Project Structure

DermaIQ/
├── app.py # Main Streamlit app
├── classification_dataset.py # Organizes raw dataset into class folders
├── data_loader.py # ImageDataGenerator loader
├── utils.py # Preprocessing & prediction utilities
├── gradcamcustomcnn.py # Grad-CAM for Custom CNN
├── burn_model_final.h5 # Trained MobileNetV2 model
├── burn_model_customcnn_functional.h5 # Trained Custom CNN model
├── training_metrics.png # Accuracy & loss graph (MobileNetV2)
├── training_metrics_custom_cnn.png # Accuracy & loss graph (Custom CNN)
├── confusion_matrix.png # Confusion matrix (MobileNetV2)
├── confusion_matrix_custom_cnn.png # Confusion matrix (Custom CNN)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DermaIQ.git
cd DermaIQ
2. Install Dependencies

pip install streamlit tensorflow opencv-python pillow numpy
3. Run the Application
streamlit run app.py
Alternatively:


python -m streamlit run app.py
📁 Dataset Preparation
Organize your dataset as follows:

E:/RECONNECT/data/
├── burn1.jpg
├── burn1.txt  # Label (0, 1, or 2)
├── burn2.jpg
├── burn2.txt
...
Now, convert it into class-wise folders by running:

python classification_dataset.py
This will create:


E:/RECONNECT/classification_dataset/
├── 0/   # First-degree burns
├── 1/   # Second-degree burns
├── 2/   # Third-degree burns
🧪 Training (Optional)
Use data_loader.py to load your dataset:

from data_loader import get_data_generators
train_gen, val_gen = get_data_generators("E:/RECONNECT/classification_dataset")
Train your models (MobileNetV2 or Custom CNN) and save them as:

burn_model_final.h5

burn_model_customcnn_functional.h5

🔬 Grad-CAM Heatmap
Grad-CAM helps visualize where the model is focusing in the image when making a prediction. It is automatically displayed in the app after a burn classification.

💬 Medical Chatbot
A built-in Gemini (Google AI) chatbot provides:

First-aid instructions

Burn care guidelines

Answers to medical queries

Access it via the third tab in the app interface.

🩺 Burn Classes
Label	Class Name	Description
0	First-degree Burn	Mild (e.g., sunburn)
1	Second-degree Burn	Blisters, moderate tissue damage
2	Third-degree Burn	Deep burns, medical emergency

📊 Evaluation Dashboard
Evaluate the performance of both models using:

📈 Accuracy & Loss Graphs

training_metrics.png

training_metrics_custom_cnn.png

📉 Confusion Matrices

confusion_matrix.png

confusion_matrix_custom_cnn.png

🙌 Team
DermaIQ Team
B.Tech (Cybersecurity & AI)
Sir Padampat Singhania University

🔗 Acknowledgements
TensorFlow / Keras

Streamlit

OpenCV

Gemini (Google AI)

Burn classification research papers and datasets
