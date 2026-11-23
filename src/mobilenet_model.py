
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_model(num_classes=9):

    # Load MobileNetV2 pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build classification model
    inputs = layers.Input(shape=(224, 224, 3))

    # Extract image features
    x = base_model(inputs, training=False)

    # Convert feature maps into a vector
    x = layers.GlobalAveragePooling2D()(x)

    # Reduce overfitting
    x = layers.Dropout(0.3)(x)

    # Classify into nine disease classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Plant_Disease_MobileNetV2"
    )

    return model


if __name__ == "__main__":

    model = build_mobilenet_model()

    model.summary()
