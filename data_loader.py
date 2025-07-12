import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, image_size=(224, 224), batch_size=32):
  
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, val_generator



if __name__ == "__main__":
    base_dir = r"E:\RECONNECT\classification_dataset"  # Your output folder with 0,1,2 folders

    train_gen, val_gen = get_data_generators(base_dir)

    print("\nClass Indices Mapping:", train_gen.class_indices)
    print(f"Total Training Images: {train_gen.samples}")
    print(f"Total Validation Images: {val_gen.samples}")
