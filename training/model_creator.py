import tensorflow as tf
from tensorflow.keras import layers, models

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.SpatialDropout2D(0.2))
    result.add(layers.ReLU())
    return result

def create_model(num_classes, image_size):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    layers_out = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers_out)

    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    up_stack = [
        upsample(512, 3, apply_dropout=True),
        upsample(256, 3, apply_dropout=True),
        upsample(128, 3),
        upsample(64, 3),
        upsample(32, 3),
    ]

    inputs = layers.Input(shape=(image_size, image_size, 3))
    skips = down_stack(inputs)
    x = skips[-1]
    skips = list(reversed(skips[:-1]))

    concat = layers.Concatenate()

    for up, skip in zip(up_stack[:-1], skips):
        x = up(x)
        x = concat([x, skip])

    x = up_stack[-1](x)

    outputs = layers.Conv2D(
        num_classes, 1, activation='softmax'
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
