
import tensorflow as tf
from pathlib import Path

from data_loader import get_datasets
from cnn_model import build_cnn_model


# --------------------------------
# 1. Training configuration
# --------------------------------

EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "cnn_best.keras"


# --------------------------------
# 2. Load preprocessed datasets
# --------------------------------

print("\nLoading datasets...")

train_ds, val_ds, test_ds, class_names = get_datasets(
    model_type="cnn"
)

print("\nClasses:", class_names)


# --------------------------------
# 3. Data augmentation
# --------------------------------

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.08),

    tf.keras.layers.RandomZoom(0.10)

], name="data_augmentation")


def augment_images(images, labels):

    images = augmentation(images, training=True)

    return images, labels


train_ds = train_ds.map(
    augment_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

print("\nData augmentation enabled.")


# --------------------------------
# 4. Build CNN model
# --------------------------------

print("\nBuilding CNN model...")

model = build_cnn_model(num_classes=len(class_names))


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

        patience=6,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=0.00001,

        verbose=1

    )

]


# --------------------------------
# 7. Train the CNN
# --------------------------------

print("\nStarting CNN training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# --------------------------------
# 8. Save final training results
# --------------------------------

model.save(MODEL_DIR / "cnn_final.keras")

print("\nCNN training completed!")

print("Best model:", BEST_MODEL_PATH)

print("Final model:", MODEL_DIR / "cnn_final.keras")
