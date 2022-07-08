import zipfile
import os
import pathlib

import tensorflow as tf
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

    # View a random image from the training dataset
    img = view_random_image(target_dir='datasets\\pizza_steak\\train\\',
                            target_class='steak')
    # plt.show()

    # View the img (actually just a big array/tensor)
    print(img)
    # View the image shape
    print(img.shape)
    # Get all the pixel values between 0 & 1
    print(img/255.)

    '''A (typical) architecture of a convolutional neural network'''

