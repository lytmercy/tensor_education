import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import random
import pathlib
import os

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


def make_cnn_model():
    model = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
        MaxPool2D(pool_size=2),  # reduce number of features by half
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model


def compile_cnn_model(model, learn_rate=0.001):
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learn_rate),
                  metrics=['accuracy'])


def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor
    adn reshapes it to (img_shape, img_shape, colour_channel)

    :param filename: image file that will be read
    :param img_shape: shape into which the file will be transformed
    :return: image instance
    """
    # Read in target file (on image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image ( to the same size our model was trained on)
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img


def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.

    :param model: pretrained model
    :param filename: image file which be plotted
    :param class_names: array with class names which `model` will be predicted
    :return: None
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


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
    # train_datagen = ImageDataGenerator(rescale=1/255.)
    # test_datagen = ImageDataGenerator(rescale=1/255.)

    # Turn it into batches
    # train_data = train_datagen.flow_from_directory(directory=train_dir,
    #                                                target_size=(224, 224),
    #                                                class_mode='binary',
    #                                                batch_size=32)
    #
    # test_data = test_datagen.flow_from_directory(directory=test_dir,
    #                                              target_size=(224, 224),
    #                                              class_mode='binary',
    #                                              batch_size=32)

    # Get a sample of the training data batch
    # images, labels = train_data.next()  # get the `next` batch of images/labels
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

    '''4. Fit a model'''

    # Check lengths of training and test data generators
    # print((len(train_data), len(test_data)))

    # Fit the model
    # history_4 = model_4.fit(train_data,
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data),
    #                         validation_data=test_data,
    #                         validation_steps=len(test_data))

    '''5. Evaluate the model'''

    # Plot the training curves
    # pd.DataFrame(history_4.history).plot(figsize=(10, 7))
    # plt.show()

    # === Make function for plot the validation and training data separately above ^ ===

    # Use function
    # plot_loss_curves(history_4)
    # plt.show()

    # model_4.summary()

    '''6. Adjust the model parameters'''

    # Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
    model_5 = make_cnn_model()

    # Compile model (same as model_4)
    compile_cnn_model(model_5)

    # history_5 = model_5.fit(train_data,
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data),
    #                         validation_data=test_data,
    #                         validation_steps=len(test_data))

    # model_5.summary()

    # Plot loss curves of model_5 results
    # plot_loss_curves(history_5)
    # plt.show()

    # Create ImageDataGenerator training instance with data augmentation
    train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                                 rotation_range=20,  # rotate the image slightly between 0 and 20 degrees
                                                                    # (note: this is an int not a float)
                                                 shear_range=0.2,  # shear the image
                                                 zoom_range=0.2,  # zoom into the image
                                                 width_shift_range=0.2,  # shift the image width ways
                                                 height_shift_range=0.2,  # shift the image height ways
                                                 horizontal_flip=True)  # flip the image on the horizontal axis

    # Create ImageDataGenerator training instance without data augmentation
    train_datagen = ImageDataGenerator(rescale=1/255.)

    # Create ImageDataGenerator test instance without data augmentation
    test_datagen = ImageDataGenerator(rescale=1/255.)

    # Import data and augment it from training directory
    print("Augmented training images:")
    train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                       target_size=(224, 224),
                                                                       batch_size=32,
                                                                       class_mode='binary',
                                                                       shuffle=False)  # Don't shuffle for demonstration
                                                                                       # purposes, usually a good
                                                                                       # thing to shuffle

    # Create non-augmented data batches
    print("Non-augmented training images:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   shuffle=False)  # Don't shuffle for demonstration purposes

    print("Unchanged test images:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='binary')

    # Get data batch samples
    images, labels = train_data.next()
    augmented_images, augmented_labels = train_data_augmented.next()  # Note:labels aren't augmented, they stay the same

    # Show original image and augmented image
    # random_number = random.randint(0, 32)  # we're making batches of size 32, so we'll get a random instance
    # plt.imshow(images[random_number])
    # plt.title("Original image")
    # plt.axis(False)
    # plt.figure()
    # plt.imshow(augmented_images[random_number])
    # plt.title("Augmented image")
    # plt.axis(False)

    # plt.show()

    # Create the model (same as model_5)
    model_6 = make_cnn_model()

    compile_cnn_model(model_6)
    # history_6 = model_6.fit(train_data_augmented,  # changed to augmented training data
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data_augmented),
    #                         validation_data=test_data,
    #                         validation_steps=len(test_data))

    # Check model's performance history training an augmented data
    # plot_loss_curves(history_6)
    # plt.show()

    # import data and augment it from directories
    train_data_augmented_shuffled = train_datagen_augmented.flow_from_directory(train_dir,
                                                                                target_size=(224, 224),
                                                                                batch_size=32,
                                                                                class_mode='binary',
                                                                                shuffle=True)  # Shuffle data (default)

    # Create the model (same as model_5 and model_6)
    model_7 = make_cnn_model()

    compile_cnn_model(model_7)

    # Fit the model
    # history_7 = model_7.fit(train_data_augmented_shuffled,  # now the augmented data is shuffled
    #                         epochs=5,
    #                         steps_per_epoch=len(train_data_augmented_shuffled),
    #                         validation_data=test_data,
    #                         validation_steps=len(test_data))

    # Check model's performance history training on augmented data
    # plot_loss_curves(history_7)
    # plt.show()

    '''7. Repeat until satisfied'''

    # Create a CNN model (same as Tiny VGG but for binary classification - https://poloclub.github.io/cnn-explainer/)
    model_8 = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),  # same input shape as our images
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    compile_cnn_model(model_8)

    history_8 = model_8.fit(train_data_augmented_shuffled,
                            epochs=5,
                            steps_per_epoch=len(train_data_augmented_shuffled),
                            validation_data=test_data,
                            validation_steps=len(test_data))

    # Check model_1 architecture (same as model_8)
    # model_8.summary()

    # Check out the TinyVGG model performance
    # plot_loss_curves(history_8)
    # plt.show()

    '''Making a prediction with our trained model'''

    # Get the class names
    data_dir = pathlib.Path('datasets/pizza_steak/train/')  # turn our training path into a Python path
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    # Classes we're working with
    # print(class_names)

    # View our example image
    # steak = mpimg.imread('datasets\\pizza_steak\\train\\steak\\9555.jpg')
    # plt.imshow(steak)
    # plt.axis(False)
    # plt.show()

    # Check the shape of our image
    # print(steak.shape)

    # === Creating function load_and_prep_image above ^ ===

    # Load in and preprocess our custom image
    steak = load_and_prep_image("datasets\\pizza_steak\\train\\steak\\9555.jpg")
    # print(steak)

    # Make a prediction on our custom image (spoiler: this won't work)
    # model_8.predict(steak)

    # Add an extra axis
    # print(f"Shape before new dimension: {steak.shape}")
    steak = tf.expand_dims(steak, axis=0)  # add an extra dimension at axis 0
    # steak = steak[tf.newaxis, ...]  # alternative to the above, '...' is short for 'every other dimension'
    # print(f"Shape after new dimension: {steak.shape}")
    # print(steak)

    # Make a prediction on custom image tensor
    pred = model_8.predict(steak)
    print(pred)

    # Remind ourselves of our class names
    print(class_names)

    # We can index the predicted class by rounding the prediction probability
    pred_class = class_names[int(tf.round(pred)[0][0])]
    print(pred_class)

    # === Creating function pred_and_plot above ^ ===

    # Test our model on a custom image
    img_path = "datasets\\pizza_steak\\train\\steak\\9555.jpg"
    pred_and_plot(model_8, img_path, class_names)
    plt.show()

    # Download another test image and make a prediction on it
    # os.system('wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg')
    pred_and_plot(model_8, '03-pizza-dad.jpeg', class_names)
    plt.show()

