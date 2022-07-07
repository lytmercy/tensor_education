import tensorflow as tf
from keras.utils import plot_model
from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import itertools
from sklearn.metrics import confusion_matrix


def make_multiclass_model(number_units=4, dense_activation='relu', output_activation='softmax'):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer (we had to reshape 28x28 to 784,
                                                        # the "Flatten" layer does this for us)
        tf.keras.layers.Dense(number_units, activation=dense_activation),
        tf.keras.layers.Dense(number_units, activation=dense_activation),
        tf.keras.layers.Dense(10, activation=output_activation)  # output shape is 10, activation is softmax
    ])
    return model


def compile_multiclass_model(model, learn_rate=0.001):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # different loss function for
                                                                         # multiclass classification
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
                  metrics=['accuracy'])


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values will be used.

    example_usage: make_confusion_matrix(y_true=test_labels,  # ground truth test labels
                                         y_pred=y_preds,  # predicted labels
                                         classes=class_names,  # array of class label names
                                         figsize=(15, 15),
                                         text_size=10)

    :param y_true: Array of truth labels (must be same as y_pred).
    :param y_pred: Array of predicted labels (must be same shape as y_true).
    :param classes: Array of class labels (e.g. string form). If 'None', integer labels are used.
    :param figsize: Size of output figure (default=(10, 10)).
    :param text_size: Size of output figure text (default=15).
    :return: A labelled confusion matrix plot comparing y_true and y_pred.
    """
    # Create the confusion matrix
    confuse_matrix = confusion_matrix(y_true, y_pred)
    confuse_matrix_norm = confuse_matrix.astype('float') / confuse_matrix.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = confuse_matrix.shape[0]  # find number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(confuse_matrix, cmap=plt.cm.Blues)  # colours will represent how 'correct'
                                                         # a class is, darker == better
    fig.colorbar(cax)

    # There a list of classes
    if classes:
        labels = classes
    else:
        labels = np.arange(confuse_matrix.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will label with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Set the threshold for different colours
    threshold = (confuse_matrix.max() + confuse_matrix.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(confuse_matrix.shape[0]), range(confuse_matrix.shape[1])):
        plt.text(j, i, f"{confuse_matrix[i, j]} ({confuse_matrix_norm[i, j]*100:.1f}%",
                 horizontalalignment='center',
                 color='white' if confuse_matrix[i, j] > threshold else 'black',
                 size=text_size)


def plot_random_image(model, images, true_labels, classes):
    """
    Picks a random image, plots it and labels it with a predicted and truth label.

    :param model: a trained model (trained on data similar to what's in images).
    :param images: a set of random images (in tensor form).
    :param true_labels: array of ground truth labels for images.
    :param classes: array of class names for images.
    :return: A plot of random image from `images` with a predicted class label from `model`
    as well as the truth class label from `true_labels`.
    """
    # Setup random integer
    i = random.randint(0, len(images))

    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))  # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Plot the target image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the colour of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    # Add xlabel information (prediction/true label)
    plt.title("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                    100*tf.reduce_max(pred_probs),
                                                    true_label),
              color=color)  # set the color to green or red


def run():
    """02.4 Working with a larger example (multiclass classification"""

    # The data has already been sorted into  training and test sets for us
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

    # Show the first training example
    # print(f"Training sample:\n{train_data[0]}\n")
    # print(f"Training label: {train_labels[0]}")

    # Check the shape of our data
    # print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

    # Check shape of a single example
    # print(train_data[0].shape, train_labels[0].shape)

    # Plot a single example
    # plt.imshow(train_data[7])
    # plt.show()

    # Check our samples label
    # print(train_labels[7])

    # Create small list of class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # How many classes are there (this will be ouu output shape)?
    # print(len(class_names))

    # Plot on example image and its label
    # plt.imshow(train_data[17], cmap=plt.cm.binary)  # change the colours to black & white
    # plt.title(class_names[train_labels[17]])
    # plt.show()

    # Plot multiple random images of fashion MNIST
    # plt.figure(figsize=(7, 7))
    # for i in range(4):
    #     ax = plt.subplot(2, 2, i + 1)
    #     rand_index = random.choice(range(len(train_data)))
    #     plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    #     plt.title(class_names[train_labels[rand_index]])
    #     plt.axis(False)
    #
    # plt.show()

    # === Build new model_11 ===
    tf.random.set_seed(42)

    model_11 = make_multiclass_model()
    compile_multiclass_model(model_11)

    # non_norm_history = model_11.fit(train_data,
    #                                 train_labels,
    #                                 epochs=10,
    #                                 validation_data=(test_data, test_labels))  # see how the model performs
    #                                                                            # on the test set during training

    # Check the shapes of our model
    # Note: the "None" in (None, 784) is for batch_size, we'll cover this in a later module
    # model_11.summary()

    # Plot non-normalized data loss curves
    # pd.DataFrame(non_norm_history.history).plot(title="Non-normalized Data")
    # plt.show()

    # Check the min and max values of the training data
    # print(train_data.min(), train_data.max())

    # Divide train and test images by the maximum value (normalize it)
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Check the min and max values of the training data
    # print(train_data.min(), train_data.max())

    # Fit the same model (but with normalize data)
    # norm_history = model_11.fit(train_data,
    #                             train_labels,
    #                             epochs=10,
    #                             validation_data=(test_data, test_labels))

    # Plot normalized data loss curves
    # pd.DataFrame(norm_history.history).plot(title="Normalized data")
    # plt.show()

    # === Build new model_13 ===
    # tf.random.set_seed(42)
    #
    # model_13 = make_multiclass_model()
    # compile_multiclass_model(model_13)

    # Create the learning rate callback
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

    # find_lr_history = model_13.fit(train_data,
    #                                train_labels,
    #                                epochs=40,  # model already doing pretty good with current LR,
    #                                            # probably don't need 100 epochs
    #                                validation_data=(test_data, test_labels),
    #                                callbacks=[lr_scheduler])

    # lrs = 1e-3 * (10**(np.arange(40)/20))
    # plt.semilogx(lrs, find_lr_history.history['loss'])  # want the x-axis to be log-scale
    # plt.xlabel("Learning rate")
    # plt.ylabel("Loss")
    # plt.title("Finding the ideal learning rate")
    # plt.show()

    # === Build new model_14 ===
    tf.random.set_seed(42)

    model_14 = make_multiclass_model()
    compile_multiclass_model(model_14)

    history = model_14.fit(train_data,
                           train_labels,
                           epochs=20,
                           validation_data=(test_data, test_labels))

    # Creating function for make confusion matrix

    # Make predictions with the most recent model
    y_probs = model_14.predict(test_data)  # 'probs' is short for probabilities

    # View the first 5 predictions
    # print(y_probs[:5])

    # See the predicted class number and label for the first example
    # print(y_probs[0].argmax(), class_names[y_probs[0].argmax()])

    # Convert all the predictions from probabilities to labels
    y_preds = y_probs.argmax(axis=1)

    # View the first 10 prediction labels
    # print(y_preds[:10])

    # Check out the non-prettified confusion matrix
    # print(confusion_matrix(y_true=test_labels,
    #                        y_pred=y_preds))

    # Make a prettier confusion matrix
    # make_confusion_matrix(y_true=test_labels,
    #                       y_pred=y_preds,
    #                       classes=class_names,
    #                       figsize=(15, 15),
    #                       text_size=10)
    # plt.show()

    # Create a function for plotting a random image along with its prediction

    # Check out a random image as well as its prediction
    plot_random_image(model=model_14,
                      images=test_data,
                      true_labels=test_labels,
                      classes=class_names)
    plt.show()

    '''What patterns is our model learning?'''

    # Find the layers of our most recent model
    print(model_14.layers)

    # Extract a particular layer
    print(model_14.layers[1])

    # Get the patterns of a layer in our network
    weights, biases = model_14.layers[1].get_weights()

    # Shape = 1 weight matrix the size of our input data (28x28) per neuron (4)
    print(weights)
    print(weights.shape)

    # Shape = 1 bias per neuron (we use 4 neurons in the first layer)
    print(biases)
    print(biases.shape)

    # Can now calculate the number of parameters in our model
    model_14.summary()

    # See the inputs and outputs of each layer
    plot_model(model_14, show_shapes=True)
