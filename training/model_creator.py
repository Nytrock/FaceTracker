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
    # 1. Базовый encoder
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"  # transfer learning почти всегда даёт плюс
    )

    # Можно взять более богатый набор skip‑слоёв
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers_out = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers_out)

    # 2. Частично размораживаем encoder после прогрева
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # 3. Decoder
    up_stack = [
        upsample(512, 3, apply_dropout=True),   # 4x4 -> 8x8
        upsample(256, 3, apply_dropout=True),   # 8x8 -> 16x16
        upsample(128, 3),                       # 16x16 -> 32x32
        upsample(64, 3),                        # 32x32 -> 64x64
    ]

    inputs = layers.Input(shape=(image_size, image_size, 3))
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    concat = layers.Concatenate()

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    # 4. Последний upsample до исходного разрешения
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding='same', activation='relu'
    )(x)  # 64x64 -> 128x128

    # 5. Голова сегментации
    outputs = layers.Conv2D(
        num_classes, 1, activation='softmax'
    )(x)  # logits -> softmax для удобства обучения

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
