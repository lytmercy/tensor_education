import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import wget
import zipfile
import pathlib
import random

# Import function for plot random images from tensor_03_0_beginning
from src.tensor_03_computer_vision.tensor_03_0_beginning import view_random_image
# Import function for plot loss curves from tensor_03_1_binary_classification
from src.tensor_03_computer_vision.tensor_03_1_binary_classification import plot_loss_curves
# Import function for prediction and plot prediction from tensor_03_1_binary_classification
from src.tensor_03_computer_vision.tensor_03_1_binary_classification import pred_and_plot
# Import function for load and preprocess image from tensor_03_1_binary_classification
from src.tensor_03_computer_vision.tensor_03_1_binary_classification import load_and_prep_image


def multi_pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with a trained model
    and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def run():
    """03.2 Multi-class Classification"""

    '''1. Import and become one with the data'''

    # Download zip file of 10_food_classes images
    # wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip")

    # Unzip the downloaded file
    # zip_ref = zipfile.ZipFile('10_food_classes_all_data.zip', 'r')
    # zip_ref.extractall()
    # zip_ref.close()

    # Walk through 10_food_classes directory and list number of files
    # for dirpath, dirnames, filenames in os.walk('datasets\\10_food_classes_all_data'):
    #     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    train_dir = 'datasets\\10_food_classes_all_data\\train\\'
    test_dir = 'datasets\\10_food_classes_all_data\\test\\'

    # Get the class names for our multi-class dataset
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    # print(class_names)

    # View a random image from the training dataset
    # img = view_random_image(target_dir=train_dir,
    #                         target_class=random.choice(class_names))  # get a random class name
    # plt.show()

    '''2. Preprocess the data (prepare it for a model)'''

    # Rescale the data and create data generator instances
    train_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen = ImageDataGenerator(rescale=1/255.)

    # Load data in from directories and turn it into batches
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='categorical')  # changed to categorical for multi-class

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

    '''3. Create a model (start with a baseline)'''

    # Create our model (a clone of model_8, except to be multi-class)
    model_9 = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')  # changed to have 10 neurons (same as number of classes) and
                                         # 'softmax' activation function
    ])

    # Compile the model
    model_9.compile(loss='categorical_crossentropy',  # changed to categorical_crossentropy
                    optimizer=Adam(),
                    metrics=['accuracy'])

    '''4. Fit the model'''

    # history_9 = model_9.fit(train_data,
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data),
    #                         validation_data=test_data,
    #                         validation_steps=len(test_data))

    '''5. Evaluate the model'''

    # model_9.evaluate(test_data)

    # Check out hte model's loss curves on the 10 classes of data
    # (note: this function comes from tensor_03_1_binary_classification)
    # plot_loss_curves(history_9)
    # plt.show()

    '''6. Adjust the model parameters'''
    """
    So our next steps will be to try and prevent our model overfitting. A couple of ways to prevent overfitting include:
    - Get more data - Having more data gives the model more opportunities to learn patterns, 
    patterns which may be more generalizable to new examples.
    - Simplify model - If the current model is already overfitting the training data, it may be too complicated of 
    a model. This means it's learning the patterns of the data to well and isn't able to generalize well to unseen data.
    One way to simplify a model is to reduce the number of layers it uses or to reduce the number of hidden units 
    in each layer.
    - Use data augmentation - Data augmentation manipulates the training data in a way so that's harder for the model to
    learn as it artificially adds more variety to the data. If a model is able to learn patterns in augmented data, the 
    model may be able to generalize better to unseen data.
    - Use transfer learning - Transfer learning involves leverages the patterns (also called pretrained weights) one
    model has learned to use as the foundation for your own task. In our case, we could use one computer vision model
    pretrained on a large variety of images and then tweak it slightly to be more specialized for food images.
    
    Note: Preventing overfitting is also referred to as regularization.
    """

    # Try a simplified (removed two layers)
    model_10 = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
        MaxPool2D(),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    model_10.compile(loss='categorical_crossentropy',
                     optimizer=Adam(),
                     metrics=['accuracy'])

    # history_10 = model_10.fit(train_data,
    #                           epochs=5,
    #                           steps_per_epoch=len(train_data),
    #                           validation_data=test_data,
    #                           validation_steps=len(test_data))

    # Check out the loss curves of model_10
    # plot_loss_curves(history_10)
    # plt.show()

    # Create augmented data generator instance
    train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                                 rotation_range=20,
                                                 width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 zoom_range=0.2,
                                                 horizontal_flip=True)

    train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                       target_size=(224, 224),
                                                                       batch_size=32,
                                                                       class_mode='categorical')

    # Clone the model (use the same architecture)
    model_11 = tf.keras.models.clone_model(model_10)

    # Compile the cloned model (same setup as used for model_10)
    model_11.compile(loss='categorical_crossentropy',
                     optimizer=Adam(),
                     metrics=['accuracy'])

    history_11 = model_11.fit(train_data_augmented,
                              epochs=5,
                              steps_per_epoch=len(train_data_augmented),
                              validation_data=test_data,
                              validation_steps=len(test_data))

    # Check out our model's performance with augmented data
    # plot_loss_curves(history_11)
    # plt.show()

    '''Making a prediction with our trained model'''

    # What classes has our model been trained on?
    print(class_names)

    # download samples
    # imgs = ["https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg",
    #         "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg",
    #         "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-hamburger.jpeg",
    #         "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-sushi.jpeg"]

    # wget.download(imgs[0])
    # wget.download(imgs[1])
    # wget.download(imgs[2])
    # wget.download(imgs[3])

    # Make a prediction using model_11
    # pred_and_plot(model=model_11,
    #               filename='03-steak.jpeg',
    #               class_names=class_names)
    # plt.show()
    # pred_and_plot(model=model_11,
    #               filename='03-sushi.jpeg',
    #               class_names=class_names)
    # plt.show()
    # pred_and_plot(model=model_11,
    #               filename='03-pizza-dad.jpeg',
    #               class_names=class_names)
    # plt.show()

    # Load in and preprocess our custom image
    img = load_and_prep_image('03-steak.jpeg')

    # Make a prediction
    pred = model_11.predict(tf.expand_dims(img, axis=0))

    # Match the prediction class to the highest prediction probability
    # pred_class = class_names[pred.argmax()]
    # plt.imshow(img)
    # plt.title(pred_class)
    # plt.axis(False)
    # plt.show()

    # Check the output of the predict function
    # print(pred)

    # Find the predicted class name
    # print(class_names[pred.argmax()])

    # === Adjust function to work with multi-class above ^ ===

    # use function
    # multi_pred_and_plot(model_11, '03-steak.jpeg', class_names)
    # plt.show()
    # multi_pred_and_plot(model_11, '03-sushi.jpeg', class_names)
    # plt.show()
    # multi_pred_and_plot(model_11, '03-pizza-dad.jpeg', class_names)
    # plt.show()
    # multi_pred_and_plot(model_11, '03-hamburger.jpeg', class_names)
    # plt.show()

    '''Saving and loading our model'''

    # Save a model
    model_11.save('saved_trained_model')

    # Load in a model and evaluate it
    loaded_model_11 = tf.keras.models.load_model('saved_trained_model')
    print(f"Loaded model result:\n{loaded_model_11.evaluate(test_data)}")

    # Compare our unsaved model's results (same as above)
    print(f"Existing model result:\n{model_11.evaluate(test_data)}")
