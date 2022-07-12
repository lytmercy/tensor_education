import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import function for plotting loss curves from tensor_03_1_binary_classification
from tensor_03_computer_vision.tensor_03_1_binary_classification import plot_loss_curves
# Import function for plotting image and it's prediction from tensor_03_1_binary_classification
from tensor_03_computer_vision.tensor_03_1_binary_classification import pred_and_plot


def run():
    """03.3 Exercises

    1. Spend 20-minutes reading and interacting with the CNN explainer website.
        - What are the key terms? e.g. explain convolution in your own words, pooling in your own words

    2. Play around with the "understanding hyperparameters" section in the CNN explainer website for 10-minutes.
        - What is the kernel size?
        - What is the stride?
        - How could you adjust each of these in TensorFlow code?

    3. Take 10 photos of two different things and build your own CNN image classifier
        using the techniques we've built before.

    4. Find an ideal learning rate for a simple convolutional neural network model on your 10-class dataset.
    """

    '''Exercise - 1
    Key terms:
    - Convolutional Layer
    - Kernel (weights kernel)
    - Activation map
    - Padding, Kernel size, Stride
    - Pooling Layer
    
    Convolution - it recognises patterns in data (images).
    Pooling - it's a feature reduction, which can be learned in data.
    
    '''

    '''Exercise - 2
    - Kernel size - it's the size of a weight matrix which slides over and extracts the pattern from data.
    - Stride - it's number of cell which Kernel (weight matrix) walk (shifted) while extracting the pattern from data.
    
    Kernel size - can adjust in Conv2D() Keras layer, that is the second parameter.
    Stride - can adjust in Conv2D() Keras layer, that is the third parameter.
    '''

    '''Exercise - 3'''

    # Define training and test directory paths
    train_dir = 'datasets\\own_binary_dataset\\train\\'
    test_dir = 'datasets\\own_binary_dataset\\test\\'

    # Define class name
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(class_names)

    # Create train and test data generators and rescale the data
    train_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    # Turn it into batches
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(1080, 1080),
                                                   batch_size=2,
                                                   class_mode='binary')

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(1080, 1080),
                                                 batch_size=1,
                                                 class_mode='binary')

    # Get a sample of the training data batch
    # images, labels = train_data.next()
    # print(f"label: {labels[0]}\n{images[0]}")

    # View the first batch of labels
    # print(labels)

    # === Build binary_model ===
    binary_model = Sequential([
        Conv2D(10, 4, activation='relu', input_shape=(1080, 1080, 3)),
        Conv2D(10, 4, activation='relu'),
        Conv2D(10, 4, activation='relu'),
        MaxPool2D(pool_size=4),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=2),
        Conv2D(5, 2, activation='relu'),
        Conv2D(5, 2, activation='relu'),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    binary_model.compile(loss='binary_crossentropy',
                         optimizer=Adam(learning_rate=0.003),
                         metrics=['accuracy'])

    # binary_history = binary_model.fit(train_data,
    #                                   epochs=10,
    #                                   steps_per_epoch=len(train_data),
    #                                   validation_data=test_data,
    #                                   validation_steps=len(test_data))

    # Plot the training curves
    # plot_loss_curves(binary_history)
    # plt.show()

    # Print model summary
    # binary_model.summary()

    # Evaluate model
    # print(binary_model.evaluate(test_data))
    #
    # pred_and_plot(binary_model, '03-cube.jpg', class_names, 1080)
    # plt.show()
    # pred_and_plot(binary_model, '03-tree.jpg', class_names, 1080)
    # plt.show()

    '''Exercise - 4'''

    # Define training and test directory paths
    train_dir = 'datasets\\own_multi_class_dataset\\train\\'
    test_dir = 'datasets\\own_multi_class_dataset\\test\\'

    # Define class name
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    # print(class_names)

    train_datagen = ImageDataGenerator(rescale=1/255.,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # Turn image into batches
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(1080, 1080),
                                                   batch_size=2,
                                                   class_mode='categorical')
    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(1080, 1080),
                                                 batch_size=2,
                                                 class_mode='categorical')

    # Get a sample of the training data batch
    # images, labels = train_data.next()
    # print(f"label: {labels[0]}\n{images[0]}")

    # View the first batch of labels
    # print(labels)

    # === Build multi_class_model ===
    multi_class_model = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(1080, 1080, 3)),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=4),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=3),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    multi_class_model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(learning_rate=0.022),
                              metrics=['accuracy'])

    # Create the learning rate callback
    lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

    multi_class_history = multi_class_model.fit(train_data,
                                                epochs=40,
                                                steps_per_epoch=len(train_data),
                                                validation_data=test_data,
                                                validation_steps=len(test_data),
                                                callbacks=[lr_scheduler])

    lrs = 1e-3 * (10**(np.arange(40)/20))
    plt.semilogx(lrs, multi_class_history.history['loss'])
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.title('Finding ideal learning rate')
    plt.show()

    # Plot the training curves
    plot_loss_curves(multi_class_history)
    plt.show()

    # Print model summary
    multi_class_model.summary()
