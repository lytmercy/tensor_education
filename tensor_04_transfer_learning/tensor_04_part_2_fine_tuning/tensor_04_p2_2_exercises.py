import tensorflow as tf
from keras.applications import EfficientNetB0
from keras import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight

import numpy as np
import matplotlib.pyplot as plt

import random
import pathlib

# Import helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, pred_and_plot, compare_histories

# Define global variable
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def visualize_and_pred(model, dataset, classes):
    """Visualize any one random image from dataset and make prediction on it using a trained model.

    :param model: a some trained model for making predictions.
    :param dataset: some dataset for visualizing and getting images.
    :param classes: array of class names for images.
    :return: plot with one random image and his prediction as title (with ground truth)
    """
    # Get images and labels from dataset
    dataset_size = len(dataset)
    batch_iter = dataset.__iter__()

    # Setup random integer
    i_batch = random.randint(0, dataset_size)
    i_array = random.randint(0, BATCH_SIZE)

    image, label = None, None

    for i in range(0, dataset_size-1):
        if i == i_batch:
            image, label = batch_iter.next()
            break
        else:
            batch_iter.next()

    # Create predictions and targets
    target_image = image[i_array]
    pred_probs = model.predict(tf.expand_dims(target_image, axis=0))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[label[i_array].numpy().argmax()]

    # Plot the target image
    plt.imshow(target_image/255.)

    # Change the colour of the titles depending on if the is right or wrong
    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    # Add title (xlabel) information (prediction/true label)
    plt.title("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                    100*tf.reduce_max(pred_probs),
                                                    true_label),
              color=color)


def run():
    """04.p2.2 Exercises

    1. Write a function to visualize an image from any dataset (train or test file) and any class (e.g. "steak",
    "pizza"... etc), visualize it and make a prediction on it using a trained model.
    2. Use feature-extraction to train a transfer learning model on 10% of the Food Vision data for 10 epochs using
    tf.keras.applications.EfficientNetB0 as the base model. Use the ModelCheckpoint callback to save the weights to file.
    3. Fine-tune the last 20 layers of the base model you trained in 2 for another 10 epochs. How did it go?
    4. Fine-tune the last 30 layers of the base model you trained in 2 for another 10 epochs. How did it go?
    """

    '''Exercise - 1'''

    # === Creating function above ^ ===

    # Getting Data
    train_dir = 'datasets\\10_food_classes_all_data\\train\\'
    test_dir = 'datasets\\10_food_classes_all_data\\test\\'

    train_data = image_dataset_from_directory(train_dir,
                                              label_mode='categorical',
                                              image_size=IMG_SIZE)

    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='categorical',
                                             image_size=IMG_SIZE)

    data_dir = pathlib.Path('datasets\\10_food_classes_all_data\\train\\')
    class_name = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(class_name)

    # === Build model ===

    input_shape = (224, 224, 3)

    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomHeight(0.2),
        RandomWidth(0.2)
    ], name='data_augmentation')

    inputs = Input(shape=input_shape, name='input_layer')

    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
    own_model_1 = Model(inputs, outputs)

    base_model.trainable = True

    # Freeze all layers except for the
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Compile
    own_model_1.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.0001),
                        metrics=['accuracy'])

    # Creating a ModelCheckpoint callback
    checkpoint_path = 'checkpoints\\own_best_model\\checkpoint.ckpt'

    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          save_weights_only=True,
                                          save_best_only=True,
                                          save_freq='epoch',
                                          verbose=1)

    print("\n\nFitting Own Model 1")
    epochs = 5
    # own_history = own_model_1.fit(train_data,
    #                               epochs=epochs,
    #                               validation_data=test_data,
    #                               validation_steps=int(0.3 * len(test_data)),
    #                               callbacks=[create_tensorboard_callback('tensorflow_hub',
    #                                                                      'ownbestnetV1'),
    #                                          checkpoint_callback])

    # call function for visualize and pred
    # visualize_and_pred(own_model_1, train_data, class_name)
    # plt.show()

    '''Exercise - 2'''

    # Getting data
    train_10_percent_dir = 'datasets\\10_food_classes_10_percent\\train\\'
    test_10_percent_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    train_10_percent_data = image_dataset_from_directory(train_10_percent_dir,
                                                         label_mode='categorical',
                                                         image_size=IMG_SIZE)

    test_10_percent_data = image_dataset_from_directory(test_10_percent_dir,
                                                        label_mode='categorical',
                                                        image_size=IMG_SIZE)

    # === Building feature extraction model ===
    feature_extract_model = EfficientNetB0(include_top=False)
    feature_extract_model.trainable = False

    inputs = Input(shape=input_shape, name='input_layer')

    x = feature_extract_model(inputs)
    x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)

    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    own_model_2 = Model(inputs, outputs)

    own_model_2.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.001),
                        metrics=['accuracy'])

    # print("\n\nFitting Own Model 2")
    initial_epoch = 10
    own_feature_history = own_model_2.fit(train_10_percent_data,
                                          epochs=initial_epoch,
                                          validation_data=test_data,
                                          validation_steps=int(0.25 * len(test_data)),
                                          callbacks=[create_tensorboard_callback('tensorflow_hub',
                                                                                'ownbestnetV2'),
                                                     checkpoint_callback])

    '''Exercise - 3'''

    feature_extract_model.trainable = True

    # Freeze all layers except for the
    for layer in feature_extract_model.layers[:-20]:
        layer.trainable = False

    # Recompile the model
    own_model_2.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.0001),
                        metrics=['accuracy'])

    fine_tune_epochs_1 = initial_epoch + 10

    # Refit the model
    history_fine_tune_20_layers = own_model_2.fit(train_10_percent_data,
                                                  epochs=fine_tune_epochs_1,
                                                  validation_data=test_data,
                                                  initial_epoch=own_feature_history.epoch[-1],
                                                  validation_steps=int(0.25 * len(test_data)),
                                                  callbacks=[create_tensorboard_callback('tensorflow_hub',
                                                                                          'ownbestnetV2_1'),
                                                             checkpoint_callback])

    '''Exercise - 4'''

    feature_extract_model.trainable = True

    for layer in feature_extract_model.layers[:-30]:
        layer.trainable = False

    own_model_2.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.0001),
                        metrics=['accuracy'])

    fine_tune_epochs_2 = fine_tune_epochs_1 + 10

    history_fine_tune_30_layers = own_model_2.fit(train_10_percent_data,
                                                  epochs=fine_tune_epochs_2,
                                                  validation_data=test_data,
                                                  initial_epoch=history_fine_tune_20_layers.epoch[-1],
                                                  validation_steps=int(0.25 * len(test_data)),
                                                  callbacks=[create_tensorboard_callback('tensorflow_hub',
                                                                                         'ownbestnetV2_2')])

    compare_histories(own_feature_history, history_fine_tune_20_layers, initial_epochs=10)
    plt.show()

    compare_histories(history_fine_tune_20_layers, history_fine_tune_30_layers, initial_epochs=20)
    plt.show()
