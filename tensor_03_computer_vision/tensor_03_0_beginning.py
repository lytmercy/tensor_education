import zipfile
import os
import pathlib

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def view_random_image(target_dir, target_class):
    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + '\\' + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')

    print(f"Image shape: {img.shape}")  # show the shape of the image

    return img


def run():
    """03.0 Beginning"""

    '''Get the data'''
    # Download zip file of pizza_steak images
    # os.system("wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip")

    # Unzip the downloaded file
    # zip_ref = zipfile.ZipFile('pizza_steak.zip', 'r')
    # zip_ref.extractall()
    # zip_ref.close()

    '''Inspect the data (become one with it)'''

    # Let's check file structure
    # os.system('ls datasets/pizza_steak/')
    # os.system('ls datasets/pizza_steak/train/')
    # os.system('ls datasets/pizza_steak/train/steak/')

    # Walk through pizza_steak directory and list number of files
    # for dirpath, dirnames, filenames in os.walk('datasets/pizza_steak/'):
    #     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    # Another way to find out how many images are in a file
    num_steak_images_train = len(os.listdir('datasets/pizza_steak/train/steak'))
    # print(num_steak_images_train)

    # Get the class names (programmatically, this is much more helpful with a longer list of classes)
    data_dir = pathlib.Path('datasets/pizza_steak/train/')  # turn our training path into a Python path
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))  # created a list of class_names
                                                                                # from the subdirectories
    # print(class_names)

    # === Creating function for view a random image from the training dataset ===

    # View a random image from the training dataset
    img = view_random_image(target_dir='datasets\\pizza_steak\\train\\',
                            target_class='steak')
    # plt.show()

    # View the img (actually just a big array/tensor)
    # print(img)
    # View the image shape
    # print(img.shape)
    # Get all the pixel values between 0 & 1
    # print(img/255.)

    '''An end-to-end example'''

    tf.random.set_seed(42)

    # Preprocess data (get all the pixel values between 1 and 0, also called scaling/normalization)
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Setup the train and test directories
    train_dir = 'datasets\\pizza_steak\\train\\'
    test_dir = 'datasets\\pizza_steak\\test\\'

    # Import data from directories and turn it into batches
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   batch_size=32,  # number of images to process at a time
                                                   target_size=(224, 224),  # convert all images to be 224 x 224
                                                   class_mode='binary',  # type of problem we're working on
                                                   seed=42)

    valid_data = valid_datagen.flow_from_directory(test_dir,
                                                   batch_size=32,
                                                   target_size=(224, 224),
                                                   class_mode='binary',
                                                   seed=42)

    # Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/ )
    model_1 = Sequential([
        Conv2D(filters=10,
               kernel_size=3,
               activation='relu',
               input_shape=(224, 224, 3)),  # first layer specifies input shape (height, width, colour channels)
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=2,  # pool_size can also be (2, 2)
                  padding='valid'),  # padding can also be 'same'
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),  # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
        MaxPool2D(2),
        Flatten(),
        Dense(1, activation='sigmoid')  # binary activation output
    ])

    model_1.compile(loss='binary_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    # history_1 = model_1.fit(train_data,
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data),
    #                         validation_data=valid_data,
    #                         validation_steps=len(valid_data))

    # Check out the layers in our model
    # model_1.summary()

    '''Using the same model as before'''

    # === Build new model_2 ===
    tf.random.set_seed(42)

    model_2 = Sequential([
        Flatten(input_shape=(224, 224, 3)),  # dens layers expect a 1-dimensional vector as input
        Dense(4, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_2.compile(loss='binary_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    # history_2 = model_2.fit(train_data,
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data),
    #                         validation_data=valid_data,  # use same validation data created above
    #                         validation_steps=len(valid_data))

    # model_2.summary()

    # === Build new model_3 ===
    tf.random.set_seed(42)

    model_3 = Sequential([
        Flatten(input_shape=(224, 224, 3)),  # dense layers expect a 1-dimensional vector as input
        Dense(100, activation='relu'),  # increase number of neurons from 4 to 100 ( for each layer)
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),  # add an extra layer
        Dense(1, activation='sigmoid')
    ])

    model_3.compile(loss='binary_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    history_3 = model_3.fit(train_data,
                            epochs=5,
                            steps_per_epoch=len(train_data),
                            validation_data=valid_data,
                            validation_steps=len(valid_data))

    model_3.summary()
