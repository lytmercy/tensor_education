import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import function for tensorboard callbacks
from tensor_04_transfer_learning.tensor_04_part_1_feature_extraction.\
    tensor_04_p1_0_beginning import create_tensorboard_callback
# Import function for pot validation and training data separately
from tensor_03_computer_vision.tensor_03_1_binary_classification import plot_loss_curves


# Define GLOBAL variables
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32


def create_model(model_url, num_classes=10):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    :param model_url: (str) A TensorFlow Hub feature extraction URL.
    :param num_classes: (int) Number of output neurons in output layer,
    should be equal to number of target classes, default 10.
    :return: An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,  # freeze the underlying patterns
                                             name='feature_extraction_layer',
                                             input_shape=IMAGE_SHAPE+(3,))  # define the input image shape

    # Create our own model
    model = Sequential([
        feature_extractor_layer,  # use the feature extraction layer as the base
        Dense(num_classes, activation='softmax', name='output_layer')  # create our own output layer
    ])
    return model


def run():
    """04.p1.1 Using TensorFlow Hub"""

    train_dir = 'datasets\\10_food_classes_10_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    train_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

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

    '''Creating model using TensorFlow Hub'''

    # Restnet 50 V2 feature vector
    resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

    # Original: EfficientNetB0 feature vector (version 1)
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

    # # New: EfficientNetB0 feature vector (version 2)
    # efficientnet_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"

    # === Creating function create_model() for compare two models above ^ ===

    # === Build resnet_model ===
    resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

    # Compile
    resnet_model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(),
                         metrics=['accuracy'])

    # Fit the model
    resnet_history = resnet_model.fit(train_data_10_percent,
                                      epochs=5,
                                      steps_per_epoch=len(train_data_10_percent),
                                      validation_data=test_data,
                                      validation_steps=len(test_data),
                                      # Add TensorBoard callback to model (callback parameter takes a list)
                                      callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',  # save experiment logs here
                                                                             experiment_name='resnet50V2')])  # name of log files

    # Plot the validation and training data separately
    # plot_loss_curves(resnet_history)
    # plt.show()

    # resnet_model.summary()

    # === Build efficientnet_model ===
    efficientnet_model = create_model(model_url=efficientnet_url,  # use EfficientNetB0 TensorFlow Hub URL
                                      num_classes=train_data_10_percent.num_classes)

    # Compile
    efficientnet_model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(),
                               metrics=['accuracy'])

    # Fit the model
    efficientnet_history = efficientnet_model.fit(train_data_10_percent,
                                                  epochs=5,
                                                  steps_per_epoch=len(train_data_10_percent),
                                                  validation_data=test_data,
                                                  validation_steps=len(test_data),
                                                  callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',
                                                                                         # Track logs under different
                                                                                         # experiment name
                                                                                         experiment_name='efficientnetB0')])

    # Plot the validation and training data separately
    # plot_loss_curves(efficientnet_history)
    # plt.show()

    # efficientnet_model.summary()

    '''Comparing models using TensorBoard'''

    # # Uploading experiments to TensorBoard

    # Writing this script in console
    # tensorboard dev upload --logdir ./tensorflow_hub/ \
    #   --name "EfficientNetB0 vs. ResNet50V2" \
    #   --description "Comparing two different TF Hub feature extraction" \
    #   " models architectures using 10% of training images" \
    #   --one_shot

    '''Listing experiments you've saved to TensorBoard'''

    # Writing next script in console
    # tensorboard dev list

    '''Deleting experiments from TensorBoard'''

    # Writing next script in console
    # tensorboard dev delete --experiment_id `experiment id`
    # Result:
    # (venv) PS D:\.main\.code\.tensor_edu> tensorboard dev delete --experiment_id yXdLQyz6RLOA615mK2G9RQ
    # Deleted experiment yXdLQyz6RLOA615mK2G9RQ.
