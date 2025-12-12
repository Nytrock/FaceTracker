import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.data import AUTOTUNE


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()

        self.flip_inputs = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.flip_labels = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.brightness = tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0.0, 1.0), seed=seed)
        self.contrast = tf.keras.layers.RandomContrast(factor=0.1, seed=seed)

    def flip_aug(self, images, masks):
        images = self.flip_inputs(images)
        masks = self.flip_labels(masks)
        return images, masks

    def color_aug(self, images):
        images = self.brightness(images)
        images = self.contrast(images)
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images

    def call(self, inputs, labels):
        inputs, labels = self.flip_aug(inputs, labels)
        inputs = self.color_aug(inputs)
        return inputs, labels

def display_dataset_example(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def load_images(image_path, mask_path, image_size):
    input_image = tf.io.read_file(image_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_image = tf.image.resize(input_image, (image_size, image_size))

    input_mask = tf.io.read_file(mask_path)
    input_mask = tf.image.decode_png(input_mask, channels=1)
    input_mask = tf.image.resize(input_mask, (image_size, image_size))
    input_mask = tf.cast(input_mask, tf.int32)

    return input_image, input_mask


def get_datasets(image_size, batch_size, use_mine_annotations=False):
    images_dir = '../dataset/main/images'
    masks_dir = '../dataset/main/annotations_original'
    if use_mine_annotations:
        masks_dir = '../dataset/main/annotations_mine'

    image_files = tf.data.Dataset.list_files(os.path.join(images_dir, '*.png'), shuffle=False)
    mask_files = tf.data.Dataset.list_files(os.path.join(masks_dir, '*.png'), shuffle=False)
    dataset_size = len(image_files)

    dataset = tf.data.Dataset.zip((image_files, mask_files))
    dataset = dataset.map(lambda image, mask: load_images(image, mask, image_size), num_parallel_calls=AUTOTUNE)

    train_size = int(dataset_size * 0.8)
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    train_dataset = (train_dataset
                     .cache()
                     .shuffle(1000)
                     .batch(batch_size)
                     .map(Augment(), num_parallel_calls=AUTOTUNE)
                     .prefetch(buffer_size=AUTOTUNE))

    validation_dataset = (validation_dataset
                          .batch(batch_size)
                          .prefetch(buffer_size=AUTOTUNE))

    return train_dataset, validation_dataset
