from __future__ import annotations

import tensorflow as tf


def build_model(image_size: int, num_classes: int) -> tf.keras.Model:
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model
