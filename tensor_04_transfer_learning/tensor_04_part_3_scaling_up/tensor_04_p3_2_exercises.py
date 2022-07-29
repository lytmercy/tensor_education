import tensorflow as tf
from keras.models import Sequential, load_model
from keras import Model, mixed_precision
from keras.applications import EfficientNetB0
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
from keras.optimizers import Adam
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import random
import os

# Import helper functions
from helper_functions import plot_loss_curves, compare_histories, load_and_prep_image

# Import class for get class names
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_0_beginning import GettingData

# Import function for make predictions and visualize it
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import file_visualise_and_predict
# Import function for build and compile model
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import create_model
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import compile_model


def training_model(model, train_dataset, test_dataset, train_epochs=10):
    return model.fit(train_dataset,
                     epochs=train_epochs,
                     validation_data=test_dataset,
                     validation_steps=int(0.15 * len(test_dataset)))


def run():
    """04.p3.2 Exercises

    1. Take 3 of your own photos of food and use the trained model to make predictions on them.
    2. Train a feature-extraction transfer learning model for 10 epochs on the same data and compare its performance
    versus a model which used feature extraction for 5 epochs and fine-tuning for 5 epochs (like we've used in this
    notebook). Which method is better?
    3. Recreate our first model (the feature extraction model) with mixed_precision turned on.
        - Does it make the model train faster?
        - Does it effect the accuracy or performance of our model?
        - What's the advantages of using mixed_precision training?

    """

    '''Exercise - 1'''

    # Loading model from storage
    model = load_model('models\\101_food_class_10_percent_saved_big_dog_model')

    # Prepare own photos
    own_food_images = ['datasets\\own_food_images\\' + img_path
                       for img_path in os.listdir('datasets\\own_food_images')]
    # print(own_food_images)

    get_data = GettingData()

    class_names = get_data.get_class_names()

    # Make predictions on some my own chosen images of food
    # file_visualise_and_predict(model, own_food_images, class_names)

    '''Exercise - 2'''

    # Prepare train and test data
    train_data = get_data.get_train_data()
    test_data = get_data.get_test_data()

    num_classes = len(train_data.class_names)

    # === Build feature-extraction model ===

    feature_model, _ = create_model(num_classes)

    compile_model(feature_model)

    # feature_history = training_model(feature_model, train_data, test_data)

    # Evaluate feature-extraction model
    # print(f"Feature-extraction model results: {feature_model.evaluate(test_data)}")
    #
    # plot_loss_curves(feature_history)
    # plt.show()

    # === Build fine-tuning model ===
    # fine_model, base_model = create_model(num_classes)
    #
    # compile_model(fine_model)
    #
    # pre_fine_history = training_model(fine_model, train_data, test_data, train_epochs=5)
    #
    # base_model.trainable = True
    # for layer in base_model.layers[:-10]:
    #     layer.trainable = False
    #
    # compile_model(fine_model, 1e-4)

    # fine_history = fine_model.fit(train_data,
    #                               epochs=10,
    #                               validation_data=test_data,
    #                               validation_steps=int(.15 * len(test_data)),
    #                               initial_epoch=pre_fine_history.epoch[-1])

    # Evaluate fine-tuning model
    # print(f"Fine-tuning model results: {fine_model.evaluate(test_data)}")
    #
    # compare_histories(pre_fine_history, fine_history)

    '''In that case not the big difference between Feature-extraction (10 epochs) model and 
    Feature-extraction (5 epochs) + Fine-tuning (5 epochs) model
    Results: Feature [1.49, 0.59] vs. Fine [1.51, 0.60]
    '''

    '''Exercise - 3'''

    # === Build Feature-Extraction model with mixed_precision = True ===

    # === Build dataaugmentation layer ===
    data_augmentation = Sequential([
        RandomFlip('horizontal'),  # randomly flip images on horizontal edge
        RandomRotation(0.2),  # randomly rotate images by a specific amount
        RandomHeight(0.2),  # randomly adjust the height of an image by a specific  amount
        RandomWidth(0.2),  # randomly adjust the width of an image by a specific amount
        RandomZoom(0.2),  # randomly zoom into an image
        # Rescaling(1./255)  # keep for models like ResNet50V2, remove for EfficientNet
    ], name='data_augmentation')

    # Check policy in feature extraction model
    outputs = Dense(10, activation='softmax', name='output')
    print(f"Before outputs dtype: %s\n" % outputs.dtype)

    policy = mixed_precision.Policy('mixed_float16')
    # My GPU is does not have compute capability of at least 7.0
    # My GPU have 6.1 compute capability
    print(mixed_precision.set_global_policy('mixed_float16'))
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)

    # base_model = EfficientNetB0(include_top=False)
    # base_model.trainable = False
    #
    # inputs = Input(shape=(224, 224, 3), name='input_layer')
    # x = data_augmentation(inputs)
    # x = base_model(x, training=False)
    # x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    # x = Dense(len(train_data.class_names), name='dense_output')(x)
    # outputs = softmax(dtype='float32', name='softmax_output')(x)
    #
    # mixed_model = Model(inputs, outputs)
    #
    # print("After outputs dtype: %s" % outputs.dtype)

    '''Therefore, the implementation of this tack is in 
    Google Colab file tensor_04_p3_2_exercise_3.ipynb'''


