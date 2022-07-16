import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import random
import wget


# Import helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

# Define global variable
IMG_SIZE = (224, 224)


def run():
    """04.p2.1 Running a series of transfer learning experiments

    model_1: Use feature extraction transfer learning on 1% of the training data with data augmentation.
    model_2: Use feature extraction transfer learning on 10% of the training data with data augmentation.
    model_3: Use fine-tuning transfer learning on 10% of the training data with data augmentation.
    model_4: Use fine-tuning transfer learning on 100% of the training data with data augmentation.

    All experiments will be done using the EfficientNetB0 model within the tf.keras.applications module.
    To make sure we're keeping track of our experiments, we'll use our create_tensorboard_callback() function
    to log all of the model training logs.
    """

    # Download and unzip data
    # wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip")
    # unzip_data('10_food_classes_1_percent.zip')

    # Create training and test dirs
    train_dir_1_percent = 'datasets\\10_food_classes_1_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_1_percent\\test\\'

    # Walk through 1 percent data directory and list number of files
    walk_through_dir('datasets\\10_food_classes_1_percent')

    train_data_1_percent = image_dataset_from_directory(train_dir_1_percent,
                                                        label_mode='categorical',
                                                        batch_size=32,  # default
                                                        image_size=IMG_SIZE)

    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='categorical',
                                             batch_size=32,
                                             image_size=IMG_SIZE)

    '''Adding data augmentation right into the model'''

    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomHeight(0.2),
        RandomWidth(0.2),
        # Rescaling(1./255)  # keep for ResNet50V2, remove forEfficientNetB0
    ], name='data_augmentation')

    # View a random image
    target_class = random.choice(train_data_1_percent.class_names)  # choose a random class
    target_dir = 'datasets\\10_food_classes_1_percent\\train\\' + target_class  # create the target directory
    random_image = random.choice(os.listdir(target_dir))  # choose a random image from target directory
    random_image_path = target_dir + "/" + random_image  # create the chosen random image path
    img = mpimg.imread(random_image_path)  # read in the chosen target image
    plt.imshow(img)  # plot the target image
    plt.title(f"Original random image from class: {target_class}")
    plt.axis(False)
    plt.show()

    # Augment the image
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))  # data augmentation model requires
                                                                    # shape (None, height, width, 3)
    plt.figure()
    plt.imshow(tf.squeeze(augmented_img)/255.)  # requires normalization after augmentation
    plt.title(f"Augmented random image from class: {target_class}")
    plt.axis(False)
    plt.show()




