
import tensorflow as tf
from pathlib import Path

# --------------------------------
# 1. Configuration
# --------------------------------

DATASET_DIR = Path("dataset/processed_900")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------
# 2. Load image datasets
# --------------------------------

def load_raw_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=class_names,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=class_names,
        shuffle=False
    )

    return train_ds, val_ds, test_ds, class_names


# --------------------------------
# 3. CNN preprocessing
# --------------------------------

def prepare_for_cnn(train_ds, val_ds, test_ds):

    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(
        normalize, num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.map(
        normalize, num_parallel_calls=AUTOTUNE
    )

    test_ds = test_ds.map(
        normalize, num_parallel_calls=AUTOTUNE
    )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE)
    )


# --------------------------------
# 4. MobileNetV2 preprocessing
# --------------------------------

def prepare_for_mobilenet(train_ds, val_ds, test_ds):

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def preprocess_images(images, labels):
        images = tf.cast(images, tf.float32)
        images = preprocess(images)
        return images, labels

    train_ds = train_ds.map(
        preprocess_images, num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_images, num_parallel_calls=AUTOTUNE
    )

    test_ds = test_ds.map(
        preprocess_images, num_parallel_calls=AUTOTUNE
    )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE)
    )


# --------------------------------
# 5. Main dataset function
# --------------------------------

def get_datasets(model_type="cnn"):

    train_ds, val_ds, test_ds, class_names = (
        load_raw_datasets()
    )

    if model_type.lower() == "cnn":

        train_ds, val_ds, test_ds = prepare_for_cnn(
            train_ds, val_ds, test_ds
        )

    elif model_type.lower() == "mobilenet":

        train_ds, val_ds, test_ds = prepare_for_mobilenet(
            train_ds, val_ds, test_ds
        )

    else:
        raise ValueError(
            "model_type must be 'cnn' or 'mobilenet'"
        )

    return train_ds, val_ds, test_ds, class_names
