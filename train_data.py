# train_mobilenetv2.py

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_data_generators

# Settings
base_dir = 'classification_dataset' 
image_size = (224, 224)
batch_size = 32
num_classes = 3
epochs = 10

# Load Data
train_gen, val_gen = get_data_generators(base_dir, image_size, batch_size)

# Build Model
base_model = MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('burn_model_mobilenetv2.h5', save_best_only=True, monitor='val_accuracy', mode='max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, earlystop]
)

# Save model
model.save("burn_model_final.h5")

# ------------------- Additional Metrics & Visualizations -------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Accuracy & Loss Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

print("[INFO] Accuracy & Loss curve saved as 'training_metrics.png'.")

# Confusion Matrix & Classification Report
print("[INFO] Generating Confusion Matrix...")

val_gen.reset()
predictions = model.predict(val_gen)
pred_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

print("[INFO] Confusion matrix saved as 'confusion_matrix.png'.")

report = classification_report(true_classes, pred_classes, target_names=["First-degree", "Second-degree", "Third-degree"])
with open("classification_report.txt", "w") as f:
    f.write(report)

print("[INFO] Classification report saved as 'classification_report.txt'.")
