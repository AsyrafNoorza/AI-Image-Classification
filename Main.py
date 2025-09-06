import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# 1) Load data (expects folder structure: data/train/{class}/..., data/val/{class}/...)
IMG_SIZE = (224, 224)
BATCH = 32
train_dir = "data/train"
val_dir   = "data/valid"

train_ds = keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH, seed=42, shuffle=True
)
val_ds = keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH, seed=42, shuffle=False
)
class_names = train_ds.class_names
num_classes = len(class_names)

# 2) Cache + prefetch + basic augmentation
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# 3a) Simple CNN (quick to train)
def build_simple_cnn():
    return keras.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

# 3b) (Optional) Transfer learning (better accuracy with fewer images)
def build_transfer_model():
    base = keras.applications.EfficientNetB0(
        include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet"
    )
    base.trainable = False  # freeze for first stage
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = data_augmentation(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_transfer_model()  # or build_simple_cnn()

# 4) Compile + train
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

# (Optional) Unfreeze top layers for fine-tuning
# model.get_layer(index=??) to target blocks; here we unfreeze the base entirely:
for layer in model.layers:
    if hasattr(layer, "trainable"):
        layer.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5) Save + inference
model.save("model_tf2_image_classification.h5")

# Example single-image prediction
import numpy as np
from PIL import Image

def predict_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)[None, ...]  # (1, H, W, 3)
    # If you used the simple CNN: scale 0-1
    # If you used EfficientNet: use preprocess_input (already inside the model graph above)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    return class_names[idx], float(preds[0][idx])

label, prob = predict_image("data/test/example.jpg")
print(label, prob)
