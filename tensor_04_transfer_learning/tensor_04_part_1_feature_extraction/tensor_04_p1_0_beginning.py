import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import os
import shutil
import zipfile
import wget
import datetime


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def run():
    """04.p1.0 Beginning"""

    '''Using a GPU'''
    # os.system('nvidia-smi')

    '''Downloading and becoming one with the data'''

    # Get data (10% of labels)
    # if not(os.path.exists('10_food_classes_10_percent.zip')):
    #     wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip')

    # Unzip the downloaded file
    # zip_ref = zipfile.ZipFile('10_food_classes_10_percent.zip', 'r')
    # zip_ref.extractall('datasets\\')
    # zip_ref.close()
    # shutil.rmtree('datasets\\__MACOSX\\')

    # How many images in each folder?
    # Wolk through 10 percent data directory and list number of files
    for dirpath, dirnames, filenames in os.walk('10_food_classes_10_percent'):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    '''Creating data loaders (preparing the data)'''

    # Setup data inputs
    IMAGE_SHAPE = (224, 224)
    BATCH_SIZE = 32

    train_dir = 'datasets\\10_food_classes_10_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    train_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    print("Training images:")
    train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                              target_size=IMAGE_SHAPE,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='categorical')

    print("Testing images:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=IMAGE_SHAPE,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')

    '''Setting up callbacks (things to run whilst our model trains)'''

    # === Creating function for TensorBoard callback above ^ ===
