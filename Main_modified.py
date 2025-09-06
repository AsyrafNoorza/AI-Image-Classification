import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np
from PIL import Image
import os

# 1) Load data (modified for single folder structure)
IMG_SIZE = (224, 224)
BATCH = 32
train_dir = "data/train"
val_dir = "data/valid"
test_dir = "data/test"

# Get all image files from directories
def get_image_files(directory):
    """Get all image files from a directory"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(directory, file))
    return image_files

# Create datasets manually since we don't have class subdirectories
def create_dataset_from_files(image_files, batch_size=32):
    """Create a dataset from a list of image files"""
    def load_and_preprocess_image(file_path):
        # Load image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        
        # For people detection, we'll use a binary classification (person/no person)
        # Since all images contain people, label them as 1
        label = tf.constant(1, dtype=tf.int32)
        
        return image, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Get image files
train_files = get_image_files(train_dir)
val_files = get_image_files(val_dir)

print(f"Found {len(train_files)} training images")
print(f"Found {len(val_files)} validation images")

if len(train_files) == 0:
    print("No training images found! Please check your data/train directory.")
    exit()

# Create datasets
train_ds = create_dataset_from_files(train_files, BATCH)
val_ds = create_dataset_from_files(val_files, BATCH)

# 2) Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# 3) Build model for binary classification (person detection)
def build_people_detection_model():
    # Use a simpler, more compatible model architecture
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = data_augmentation(x)
    
    # Convolutional layers
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Binary classification: person (1) or no person (0)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_people_detection_model()

# 4) Compile + train
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)

# 5) Fine-tuning
print("Starting fine-tuning...")
for layer in model.layers:
    if hasattr(layer, "trainable"):
        layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=5,
    verbose=1
)

# 6) Save model
model.save("people_detection_model.h5")
print("Model saved as 'people_detection_model.h5'")

# 7) Test prediction function
def predict_image(path):
    """Predict if an image contains a person"""
    if not os.path.exists(path):
        return "File not found", 0.0
    
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)[None, ...]  # (1, H, W, 3)
    
    # Simple preprocessing (scale to 0-1)
    arr = arr.astype(np.float32) / 255.0
    
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    
    # Return prediction
    if idx == 1:
        return "Person detected", float(preds[0][idx])
    else:
        return "No person detected", float(preds[0][idx])

# Test with a sample image
test_files = get_image_files(test_dir)
if test_files:
    test_image = test_files[0]
    print(f"\nTesting with: {test_image}")
    label, confidence = predict_image(test_image)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
else:
    print("\nNo test images found in data/test directory")

print("\nTraining completed successfully!")
