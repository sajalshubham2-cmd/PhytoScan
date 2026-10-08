import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 8


def build_crop_disease_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = layers.Input(shape=IMG_SHAPE, name="leaf_rgb_input")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescale")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
        alpha=1.0,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap_2d")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(rate=0.35, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = layers.Dropout(rate=0.25, name="head_dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="class_probs")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="phytoscan_mobilenetv2")
    return model, base_model


def unfreeze_for_phase2(model: tf.keras.Model, base_model: tf.keras.Model) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:120]:
        layer.trainable = False
    for layer in base_model.layers[120:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
