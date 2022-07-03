import tensorflow as tf
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# Import class for init data
from tensor_01_regression.regression_data_class import RegressionData


# Create function for plotting
def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions):
    """
    Plots training data, test data and compares predictions.
    """

    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Test data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()
    plt.show()


def run():
    """01.2 Visualizing all"""

    '''Visualizing the data'''

    # Initialize class instance
    data = RegressionData()
    data.train_test_init()

    # Getting a data
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test
    # print(X_train, y_train)

    # Let's plot it with some nice colours to differentiate what's what.
    # plt.figure(figsize=(10, 7))
    # Plot training data in blue
    # plt.scatter(X_train, y_train, c='b', label='Training data')
    # Plot test data in green
    # plt.scatter(X_test, y_test, c='g', label='Testing data')
    # Show the legend
    # plt.legend()
    # plt.show()

    '''Visualizing the model'''

    # Set random seed
    tf.random.set_seed(42)

    # Create a mode
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1)
    # ])

    # Compile model
    # model.compile(loss=tf.keras.losses.mae,
    #               optimizer=tf.keras.optimizers.SGD(),
    #               metrics=["mae"])

    # Fit model
    # model.fit(X_train, y_train, epochs=100)  # commented out on purpose (not fitting it just yet)

    # Doesn't work (model not fit/built)
    # model.summary()

    # Create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])  # define the input_shape to our model
    ])

    # Compile model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])

    # This will work after specifying the input shape
    # model.summary()

    """
    Calling summary() on our model shows us the layers it contains, the output shape and the number of parameters.
    - Total params - total number of parameters in the model.
    - Trainable parameters - these are the parameters (patterns) the model can update as it trains.
    - Non-trainable parameters - these parameters aren't updated during training (this is typical when we bring
    in the already learned patterns from other models during transfer learning).
    """

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=100, verbose=0)  # verbose controls how many gets output

    # Check the model summary
    # model.summary()

    # Plotting model
    # plot_model(model, show_shapes=True)

    '''Visualizing the predictions'''

    # Make predictions
    y_preds = model.predict(X_test)

    # View the predictions
    # print(y_preds)

    # Using function for plotting prediction
    plot_predictions(X_train, y_train, X_test, y_test, y_preds)
