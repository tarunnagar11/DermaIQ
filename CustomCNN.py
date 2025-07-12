# data_loader.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        base_dir, # Use the base_dir directly, assuming it contains subdirectories for classes
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', # Set class_mode to categorical
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        base_dir, # Use the base_dir directly
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', # Set class_mode to categorical
        subset='validation',
        shuffle=True
    )

    return train_generator, val_generator

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Set parameters
image_size = (224, 224)
batch_size = 32
epochs = 25
base_dir = 'classification_dataset'  # or your local dataset path

# Load data
train_gen, val_gen = get_data_generators(base_dir, image_size=image_size, batch_size=batch_size)

# Build the custom CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 burn classes
])

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('burn_model_customcnn.h5', save_best_only=True, monitor='val_accuracy')
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, earlystop])

# Save model and history
model.save('burn_model_customcnn_final.h5')

with open('training_history_customcnn.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate and save confusion matrix/report
from sklearn.metrics import classification_report, confusion_matrix

y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

np.save('y_true_customcnn.npy', y_true)
np.save('y_pred_customcnn.npy', y_pred_classes)

report = classification_report(y_true, y_pred_classes, output_dict=True)
with open('classification_report_customcnn.pkl', 'wb') as f:
    pickle.dump(report, f)

print("✅ Custom CNN trained and evaluation data saved.")

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

# Load Sequential model
seq_model = load_model("burn_model_customcnn_final.h5")
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = seq_model(dummy_input)

# Define same architecture with Functional API (no custom layer names)
def convert_to_functional():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    return Model(inputs, outputs, name="CustomCNNFunctional")

func_model = convert_to_functional()
_ = func_model(dummy_input)

# Transfer weights safely by index
seq_layers = [l for l in seq_model.layers if l.get_weights()]
func_layers = [l for l in func_model.layers if l.get_weights()]

for seq_l, func_l in zip(seq_layers, func_layers):
    try:
        func_l.set_weights(seq_l.get_weights())
    except Exception as e:
        print(f"⚠️ Error transferring weights from {seq_l.name} → {func_l.name}: {e}")

# Save the fixed Functional model
func_model.save("burn_model_customcnn_functional.h5")
print("✅ Final functional model saved — fully Grad-CAM compatible.")

import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics_custom_cnn.png")  # ✅ Match filename used in app.py


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_gen.class_indices, yticklabels=train_gen.class_indices)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_custom_cnn.png")  # ✅ Match app.py filename
