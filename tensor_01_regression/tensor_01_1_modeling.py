import tensorflow as tf
import numpy as np


def run():
    """01.1 Modeling Machine Learning algorithm"""

    '''Steps in modeling with TensorFlow'''

    # Create features
    # X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

    # Create labels
    # y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

    # Set random seed
    tf.random.set_seed(42)

    # Create a model using the Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for 'mean absolute error'
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for 'Stochastic gradient descent'
                  metrics=["mae"])

    # Fit the model
    # model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

    # Make a prediction with the model
    # print(model.predict([17.0]))

    '''Improving a model'''

    # Fit model (this time we'll train for Longer)
    # model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

    # Try and predict what y would be if X was 17.0
    # print(model.predict([17.0]))  # the right answer is 27.0 (y = X + 10)

    '''Evaluating a model'''
    """
    For evaluating we need to visualize all:
    - The data - what data are you working with? What does it look like?
    - The model itself - what does the architecture look like? What are the different shapes?
    - The training of a model - how does a model perform while it learns?
    - The predicitons of a model - how do the predictions of a model line up 
    against the ground truth (the original labels?)
    """

    # Make a bigger dataset
    X = np.arange(-100, 100, 4)
    # print(X)

    # Make labels for the dataset (adhering to the same pattern as before)
    y = np.arange(-90, 110, 4)
    # print(y)

    # Same result as above
    y = X + 10
    # print(y)

    '''Split data into training/test set'''
    """
    Each set serves a specific purpose:
    - Training set - the model learns from this data, which is typically 70-80% of the total data available (like
    the course materials we study during the semester).
    - Validation set - the model gets tuned on this data, which is typically 10-15% of the total data available (like
    the practice exam we take before the final exam).
    - Test set - the model gets evaluated on this data to test what it has learned , it's typically 10-15% of the
    total data available (like the final exam we take at the end of the semester).
    """

    # Check how many samples we have
    print(len(X))

    # Split data into train and test sets
    X_train = X[:40]  # first 40 examples (80% of data)
    y_train = y[:40]

    X_test = X[40:]  # last 10 examples (20% of data)
    y_test = y[40:]

    print(len(X_train), len(X_test))

