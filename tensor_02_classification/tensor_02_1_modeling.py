import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def classification_model_compile_sgd(model):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])


def classification_model_compile_adam(model):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi_class
    if model.output_shape[-1] > 1:  # checks the final dimension of the model's output shape,
                                    # if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def run():
    """02.1 Modeling"""

    '''Steps in modeling'''

    # Getting data
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    # Make dataframe of features and labels
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

    # # Set random seed
    # tf.random.set_seed(42)
    #
    # # 1. Create the model using the Sequential API
    # model_1 = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1)
    # ])
    #
    # # 2. Compile the model
    # model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),  # binary since we are working with 2 classes (0 & 1)
    #                 optimizer=tf.keras.optimizers.SGD(),
    #                 metrics=['accuracy'])
    #
    # # 3. Fit the model
    # model_1.fit(X, y, epochs=5)
    #
    # # Train our model for longer (more chances to look at the data)
    # model_1.fit(X, y, epochs=200, verbose=0)
    # print(model_1.evaluate(X, y))

    # # Set random seed
    # tf.random.set_seed(42)
    #
    # # 1. Create the model (same as model_1 but with an extra layer)
    # model_2 = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1),
    #     tf.keras.layers.Dense(1)
    # ])
    #
    # classification_model_compile_sgd(model_2)
    # model_2.fit(X, y, epochs=100, verbose=0)

    # print(model_2.evaluate(X, y))

    '''Improving a model'''
    """
    Common ways to improve a deep model:
    - Adding layers
    - Increase the number of hidden units
    - Change the activation functions
    - Change the optimization function
    - Change the learning rate (in optimizer)
    - Fitting on more data
    - Fitting for longer (for more epochs)
    """

    # Set random seed
    tf.random.set_seed(42)

    # Create the model (this time 3 layers)
    model_3 = tf.keras.Sequential([
        # Before TensorFlow 2.7.0
        # tf.keras.layers.Dense(100),  # add 100 dense neurons

        # With TensorFlow 2.7.0
        # tf.keras.layers.Dense(100, input_shape=(None, 1)),  # add 100 dense neurons

        # === After TensorFlow 2.8.0 ===
        tf.keras.layers.Dense(100),  # add 100 dense neurons
        tf.keras.layers.Dense(10),  # add another layer with 10 neurons
        tf.keras.layers.Dense(1)
    ])

    classification_model_compile_adam(model_3)
    # model_3.fit(X, y, epochs=100, verbose=0)  # fit for 100 passes of the data

    # Creating function plot_decision_boundary()

    # Check out the predictions our model is making
    # plot_decision_boundary(model_3, X, y)

    # === Build regression model ===
    tf.random.set_seed(42)

    # Create some regression data
    X_regression = np.arange(0, 1000, 5)
    y_regression = np.arange(100, 1100, 5)

    # Split it into training adn test sets
    X_reg_train = X_regression[:150]
    X_reg_test = X_regression[150:]
    y_reg_train = y_regression[:150]
    y_reg_test = y_regression[150:]

    # Fit our model to the data

    # model_3.fit(tf.expand_dims(X_reg_train, axis=-1), y_reg_train, epochs=100)
    # model_3.summary()

    # We can recreate it for a regression problem
    model_3.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['mae'])

    model_3.fit(tf.expand_dims(X_reg_train, axis=-1), y_reg_train, epochs=100)

    # Make predictions with our trained model
    y_reg_preds = model_3.predict(y_reg_test)

    # Plot the model's predictions against our regression data
    plt.figure(figsize=(10, 7))
    plt.scatter(X_reg_train, y_reg_train, c='b', label='Training data')
    plt.scatter(X_reg_test, y_reg_test, c='g', label='Testing data')
    plt.scatter(X_reg_test, y_reg_preds.squeeze(), c='r', label='Predictions')
    plt.legend()
    plt.show()

