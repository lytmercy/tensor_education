import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import function view_random_image from tensor_03_0_beginning
from tensor_03_computer_vision.tensor_03_0_beginning import view_random_image


def plot_loss_curves(history):
    """
    Function for plot the validation and training data separately
    :param history: history of loss curves from fitted model.
    :return: separate loss curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()


def run():
    """03.1 Binary classification"""

    '''1. Import and become one with the data'''

    # Visualize data (requires function 'view_random_image' above)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # steak_img = view_random_image('datasets\\pizza_steak\\train\\', 'steak')
    # plt.subplot(1, 2, 2)
    # pizza_img = view_random_image('datasets\\pizza_steak\\train\\', 'pizza')
    # plt.show()

    '''2. Preprocess the data (prepare it for a model)'''

    # Define training and test directory paths
    train_dir = 'datasets\\pizza_steak\\train\\'
    test_dir = 'datasets\\pizza_steak\\test\\'

    # Create train and test data generators and rescale the data
    train_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    # Turn it into batches
    train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(224, 224),
                                                   class_mode='binary',
                                                   batch_size=32)

    test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                 target_size=(224, 224),
                                                 class_mode='binary',
                                                 batch_size=32)

    # Get a sample of the training data batch
    images, labels = train_data.next()  # get the `next` batch of images/labels
    # print(len(images), len(labels))

    # Get the first two images
    # print(images[:2], images[0].shape)

    # View the first batch of labels
    # print(labels)

    '''3. Create a model (start with a baseline)'''

    # === Build new model_4 ===
    model_4 = Sequential([
        Conv2D(filters=10,
               kernel_size=3,
               strides=1,
               padding='valid',
               activation='relu',
               input_shape=(224, 224, 3)),  # input layer (specify input shape)
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')  # output layer (specify output shape)
    ])

    model_4.compile(loss='binary_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    # Check lengths of training and test data generators
    # print((len(train_data), len(test_data)))

    # Fit the model
    history_4 = model_4.fit(train_data,
                            epochs=5,
                            steps_per_epoch=len(train_data),
                            validation_data=test_data,
                            validation_steps=len(test_data))

    # Plot the training curves
    # pd.DataFrame(history_4.history).plot(figsize=(10, 7))
    # plt.show()

    # === Make function for plot the validation and training data separately above ^ ===

    # Use function
    plot_loss_curves(history_4)
    plt.show()

    model_4.summary()
