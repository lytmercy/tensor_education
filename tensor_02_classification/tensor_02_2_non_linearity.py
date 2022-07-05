
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Import function "plot_decision_boundary()" from tensor_02_1_modeling.py
from tensor_02_classification.tensor_02_1_modeling import plot_decision_boundary


def class_compile_model(model, learn_rate=0.001):
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),  # Adam's default learning rate is 0.001
                  metrics=['accuracy'])


def run():
    """02.2 The missing piece: Non-Linearity"""

    # Getting data
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    # Make dataframe of features and labels
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

    # === Build new model ===
    tf.random.set_seed(42)

    model_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),  # 1 hidden layer with linear activation
        tf.keras.layers.Dense(1)  # output layer
    ])

    model_4.compile(loss=tf.keras.losses.binary_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # note: 'lr' used to be what was use, now
                                                                              # "learning rate"  is favoured
                    metrics=['accuracy']
                    )

    # history = model_4.fit(X, y, epochs=100, verbose=0)
    # Check out our data
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()
    # Check the decision boundary (blu is blue class, yellow is the crossover, red is red class)
    # plot_decision_boundary(model_4, X, y)

    # === Build new model_5 ===
    tf.random.set_seed(42)

    model_5 = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=tf.keras.activations.relu),  # can also do activation='relu'
        tf.keras.layers.Dense(1)  # output layer
    ])

    class_compile_model(model_5)
    # model_5.fit(X, y, epochs=100)

    # === Build new model_6 ===
    tf.random.set_seed(42)

    model_6 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),  # hidden layer-1, 4 neurons, ReLU activation
        tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),  # hidden layer-2, 4 neurons, ReLU activation
        tf.keras.layers.Dense(1)  # output layer
    ])

    class_compile_model(model_6)
    # history = model_6.fit(X, y, epochs=100, verbose=0)

    # Evaluate the model
    # print(model_6.evaluate(X, y))

    # Check out the predictions using 2 hidden layers
    # plot_decision_boundary(model_6, X, y)
    # plt.show()

    # === Build new model_7 ===
    tf.random.set_seed(42)

    model_7 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),  # hidden layer-1, RuLU activation
        tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),  # hidden layer-2, RuLU activation
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),  # output layer, sigmoid activation
    ])

    class_compile_model(model_7)
    # model_7.fit(X, y, epochs=100, verbose=0)

    # print(model_7.evaluate(X, y))
    # plot_decision_boundary(model_7, X, y)
    # plt.show()

    # Create a toy tensor (similar to the data we pass into our model)
    A = tf.cast(tf.range(-10, 10), tf.float32)
    # print(A)

    # Visualize our toy tensor
    # plt.plot(A)
    # plt.show()

    # Create sigmoid function
    def sigmoid(x):
        return 1 / (1 + tf.exp(-x))

    # Use the sigmoid function on our tensor
    # print(sigmoid(A))
    # Plot sigmoid modified tensor
    # plt.plot(sigmoid(A))
    # plt.show()

    # Create ReLU function
    def relu(x):
        return tf.maximum(0, x)

    # Pass toy tensor through ReLU function
    # print(relu(A))
    # Plot ReLU-modified tensor
    # plt.plot(relu(A))
    # plt.show()

    # Use linear function
    print(tf.keras.activations.linear(A))

    # Does the linear activation change anything?
    print(A == tf.keras.activations.linear(A))
