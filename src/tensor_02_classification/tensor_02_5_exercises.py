import tensorflow as tf
import numpy as np
import pandas as pd
import random
from keras.datasets import fashion_mnist
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Import function `plot_decision_boundary`
from src.tensor_02_classification.tensor_02_2_non_linearity import plot_decision_boundary
# Import function for visualize image and predictions from tensor_02_4_larger_example
from src.tensor_02_classification.tensor_02_4_larger_example import plot_random_image
# Import function for visualize prettier confusion matrix from tensor_02_4_larger_example
from src.tensor_02_classification.tensor_02_4_larger_example import make_confusion_matrix
# Import function for creating & compile multiclass model from tensor_02_4_larger_example
from src.tensor_02_classification.tensor_02_4_larger_example import make_multiclass_model, compile_multiclass_model


def plotting_multiply_random_images(model, images, true_labels, classes, number_images):
    """
    Picks multiply random images plots it and labels it with a predicted and truth label.

    :param model: a trained model (trained to data similar to what's in images).
    :param images: a set of random images (in tensor form).
    :param true_labels: array of ground truth labels for images.
    :param classes: array of class names for images.
    :param number_images: a number of images that will be shown.
    :return: A plot of multiply random image from `images` with a predicted class label from `model`
    as well as the truth class label from `true_labels`.
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    for i in range(number_images):
        ax = plt.subplot(2, 2, i + 1)
        plot_random_image(model, images, true_labels, classes)
        plt.axis(False)


def own_tf_softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)


def certain_class_images(model, test_images, true_labels, classes, number_images, certain_class):
    """
    Picks multiply random images of a certain class and try to predict them labels.
    Plotting all to predict and true labels.

    :param model: a trained model (trained to data similar to what's in images).
    :param test_images: a set of random images of test dataset (in tensor form).
    :param true_labels: array of ground truth test labels for test_images.
    :param classes: array of class names for test_images.
    :param number_images: a number of images that will be shown.
    :param certain_class: a certain class that will be checked when the image would be shown (in string form)
    :return: A plot of multiply random image of a certain class with predicted and true labels.
    """
    plt.figure(figsize=(8, 8))
    for i in range(number_images):
        ax = plt.subplot(2, 2, i + 1)
        while True:
            rand_index = random.choice(range(len(test_images)-1))
            if str.__contains__(classes[true_labels[rand_index]], certain_class):
                y_probs = model.predict(test_images[rand_index])
                y_preds = y_probs.argmax()
                print(tf.reduce_max(y_probs))
                true_label = classes[true_labels[rand_index]]
                pred_label = classes[y_preds]

                if true_label == pred_label:
                    color = 'green'
                else:
                    color = 'red'

                plt.imshow(test_images[rand_index], cmap=plt.cm.binary)
                plt.title("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                                100 * tf.reduce_max(y_probs),
                                                                true_label),
                          color=color)
                plt.axis(False)
                break


def run():
    """
    02.5 Exercises

    1. Play with neural networks in the TensorFlow Playground for 10-minutes. 
    Especially try different values of the learning, what happens when you decrease it? 
    What happens when you increase it? - Done!
    2. Replicate the model pictured in the TensorFlow Playground diagram below using TensorFlow code. 
    Compile it using the Adam optimizer, binary crossentropy loss and accuracy metric. 
    Once it's compiled check a summary of the model. (image = /model_from_tensorflow_playground.png)
    3. Create a classification dataset using Scikit-Learn's make_moons() function, 
    visualize it and then build a model to fit it at over 85% accuracy.
    4. Create a function (or write code) to visualize multiple image predictions for the fashion MNIST at the same time.
    Plot at least three different images and their prediction labels at the same time. Hint: 
    see the classification tutorial in the TensorFlow documentation for ideas.
    5. Recreate TensorFlow's softmax activation function in your own code. Make sure it can accept a tensor and 
    return that tensor after having the softmax function applied to it.
    6. Train a model to get 88%+ accuracy on the fashion MNIST test set. 
    Plot a confusion matrix to see the results after.
    7. Make a function to show an image of a certain class of the fashion MNIST dataset and make a prediction on it.
    For example, plot 3 images of the T-shirt class with their predictions.
    """

    '''Exercise - 2'''

    # == Building model_playground ===
    model_playground = tf.keras.Sequential([
        tf.keras.layers.Dense(6, input_shape=[2], activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_playground.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])

    model_playground.build()
    model_playground.summary()

    '''Exercise - 3'''

    # Make 1000 example
    n_samples = 1000

    # Create moons
    X, y = make_moons(n_samples, noise=0.03)

    # Check first 10 features and labels
    print(X[:10])
    print(y[:10])

    # === Build moon_model ===
    moon_model = tf.keras.Sequential([
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    moon_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                       metrics=['accuracy'])
    moon_history = moon_model.fit(X, y, epochs=20)

    pd.DataFrame(moon_history.history).plot()
    plt.show()

    # Evaluate model
    print(moon_model.evaluate(X, y))

    # Plot decision boundary
    plot_decision_boundary(moon_model, X, y)
    plt.show()

    '''Exercise - 4'''

    # === Creating a function to visualize multiple image predictions on above ^ ===

    # Get data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # === Build new model ===
    multi_model = make_multiclass_model(10)
    compile_multiclass_model(multi_model, learn_rate=0.001)
    multi_history = multi_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    # Plot loss curves
    pd.DataFrame(multi_history.history).plot(title="Loss Curves")
    plt.show()

    # Make predictions
    y_probs = multi_model.predict(X_test)
    # Convert all the predictions from probabilities to labels
    y_preds = y_probs.argmax(axis=1)

    # Use function
    plotting_multiply_random_images(multi_model, X_test, y_test, class_names, 4)
    plt.show()

    '''Exercise - 5'''

    # === Recreating `softmax` function on above ^ ===

    # Create data
    input = tf.random.normal(shape=(25, 10))
    # Use function
    outputs = own_tf_softmax(input)
    real_outputs = tf.keras.activations.softmax(input)
    # Printing outputs
    # print(outputs)
    print(tf.reduce_sum(outputs[0, :]))
    # Printing real outputs from tf.keras.activations.softmax() function
    # print(real_outputs)
    print(tf.reduce_sum(real_outputs[0, :]))

    # Checking the sums of both above softmax function equality
    sums_equal = (tf.reduce_sum(outputs[0, :]) == tf.reduce_sum(real_outputs[0, :])).__bool__()
    print(f"Sum`s equal = {sums_equal}")

    '''Exercise - 6'''

    # === Build new model ===
    accuracy_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(55, activation='relu'),
        tf.keras.layers.Dense(55, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # accuracy_model = make_multiclass_model()
    compile_multiclass_model(accuracy_model, learn_rate=0.001)
    accuracy_history = accuracy_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    # Plot loss curves
    pd.DataFrame(accuracy_history.history).plot(title="Loss Curves")
    plt.show()

    # Evaluate model
    loss, accuracy = accuracy_model.evaluate(X_test)
    print(f"Accuracy = {accuracy*100}%")

    # Summary
    accuracy_model.summary()

    # Make predictions
    y_probs = accuracy_model.predict(X_test)
    # Convert all the predictions from probabilities to label
    y_preds = y_probs.argmax(axis=1)
    # Plot a confusion matrix
    make_confusion_matrix(y_true=y_test,
                          y_pred=y_preds,
                          classes=class_names,
                          figsize=(15, 15),
                          text_size=10)
    plt.show()

    '''Exercise - 7'''

    # === Creating the function to show several images of a certain class and their prediction ===

    # Use function
    certain_class_images(accuracy_model, X_test, y_test, class_names, 4, 'T-shirt')
    plt.show()
