
import tensorflow as tf
from pathlib import Path

from data_loader import get_datasets
from mobilenet_model import build_mobilenet_model


# --------------------------------
# 1. Training configuration
# --------------------------------

EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "mobilenet_best.keras"


# --------------------------------
# 2. Load datasets
# --------------------------------

print("\nLoading MobileNetV2 datasets...")

train_ds, val_ds, test_ds, class_names = get_datasets(
    model_type="mobilenet"
)

print("\nClasses:", class_names)


# --------------------------------
# 3. Data augmentation
# --------------------------------

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.08),

    tf.keras.layers.RandomZoom(0.10)

], name="mobilenet_augmentation")


def augment_images(images, labels):

    images = augmentation(images, training=True)

    return images, labels


train_ds = train_ds.map(
    augment_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------
# 4. Build MobileNetV2
# --------------------------------

print("\nBuilding MobileNetV2 model...")

model = build_mobilenet_model(
    num_classes=len(class_names)
)


# --------------------------------
# 5. Compile model
# --------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# --------------------------------
# 6. Training callbacks
# --------------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(BEST_MODEL_PATH),

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.00001,

        verbose=1

    )

]


# --------------------------------
# 7. Train model
# --------------------------------

print("\nStarting MobileNetV2 training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# --------------------------------
# 8. Save model and history
# --------------------------------

model.save(MODEL_DIR / "mobilenet_final.keras")

history_path = Path("results")
history_path.mkdir(exist_ok=True)

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    history_path / "mobilenet_training_history.csv",
    index=False
)

print("\nMobileNetV2 training completed!")

print("Best model:", BEST_MODEL_PATH)

print("Final model:", MODEL_DIR / "mobilenet_final.keras")

print("Training history saved successfully!")
