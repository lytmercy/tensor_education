import tensorflow as tf

import tensorflow_datasets as tfds


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import scripts from helper_functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories


class Dataset:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.ds_info = None
        self.load_dir = 'datasets\\tensorflow_datasets\\food101\\'

    def load_dataset(self):
        (self.train_data, self.test_data), self.ds_info = tfds.load(name='food101',
                                                                    split=['train', 'validation'],
                                                                    shuffle_files=True,
                                                                    as_supervised=True,
                                                                    with_info=True,
                                                                    data_dir=self.load_dir)

    def preprocess_dataset(self, batch=32):
        self.train_data = self.train_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_data = self.train_data.shuffle(buffer_size=1000).batch(batch_size=batch).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.test_data = self.test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.batch(32).prefetch(tf.data.AUTOTUNE)


def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from `uint8` -> `float32` and
    reshapes image to [img_shape, img_shape, color_channels[
    """
    image = tf.image.resize(image, [img_shape, img_shape])  # reshape to img_shape
    return tf.cast(image, tf.float32), label  # return (float32_image, label) tuple


def run():
    """07.0 Preprocessing data from TensorFlow Datasets"""

    # List available datasets
    datasets_list = tfds.list_builders()  # get all available datasets in TFDS
    print('food101' in datasets_list)  # is the dataset we're after available?

    # Load in the data
    (train_data, test_data), ds_info = tfds.load(name='food101',  # target dataset to get from TFDS
                                                 # what splits of data should we get?
                                                 # note: not all datasets have train, valid, test
                                                 split=['train', 'validation'],
                                                 # shuffle files on download
                                                 shuffle_files=True,
                                                 # download data in tuple format (sample, label), e.g. (image, label)
                                                 as_supervised=True,
                                                 # include dataset metadata
                                                 # (if so, tfds.load() returns tuple (data, ds_info))
                                                 with_info=True,
                                                 # location where the dataset is saved
                                                 data_dir='datasets\\tensorflow_datasets\\food101\\')

    # Features of FOod101 TFDS
    print(ds_info.features)

    # Get class names
    class_names = ds_info.features['label'].names
    print(class_names[:10])

    '''Exploring the Food101 data from TensorFlow Datasets'''

    # Take one sample off the training data
    train_one_sample = train_data.take(1)  # samples are in format (image_tensor, label)
    # What does one sample of our training data look like?
    print(train_one_sample)

    # Output info about our training sample
    for image, label in train_one_sample:
        print(f"""
        Image shape: {image.shape}
        Image dtype: {image.dtype}
        Target class from Food101 (tensor form): {label}
        Class name (str form): {class_names[label.numpy()]}""")
        # What does an image tensor from TFDS's Food101 look like?
        print(image)

        # What are the min and max vales?
        print(tf.reduce_min(image))
        print(tf.reduce_max(image))
        '''Plot an image from TensorFLow Datasets'''
        # Plot on image tensor
        plt.imshow(image)
        plt.title(class_names[label.numpy()])  # add title to image by indexing on class_names list
        plt.axis(False)
        plt.show()

        '''Create preprocessing functions for our data'''
        # === Creating function for preprocess data above ^ ===

        # Preprocess a single sample image and check the outputs
        preprocessed_img = preprocess_img(image, label)[0]
        print(f"Image before preprocessing:\n {image[:2]}...,"
              f"\nShape: {image.shape},\nDatatype: {image.dtype}\n")
        print(f"Image after preprocessing:\n {preprocessed_img[:2]}...,"
              f"\nShape: {preprocessed_img.shape},\nDatatype: {preprocessed_img.dtype}")

        # We can still plot our preprocessed image as log as we
        # divide by 225 (for matplotlib compatibility))
        plt.imshow(preprocessed_img/255.)
        plt.title(class_names[label])
        plt.axis(False)
        plt.show()

    '''Batch & prepare datasets'''

    # Map preprocessing function to training data (and parallelize)
    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Shuffle train_data and turn it into batches and prefetch it (load it faster)
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map preprocessing function to test data
    test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    # Turn test data into batches (don't need to shuffle)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    print(train_data)
    print(test_data)

    # === Create class for getting & preprocessing data from tf datasets above ===
