import tensorflow as tf
import numpy as np
import pandas as pd

# Import class for init data
from tensor_01_regression.regression_data_class import RegressionData
# Import function for plotting predictions
from tensor_01_regression.tensor_01_2_visualizing import plot_predictions


#  Create function for MAE & MSE
def mae(y_test, y_pred):
    """
    Calculates mean absolute error between y_test and y_preds.
    """
    return tf.metrics.mean_absolute_error(y_test, y_pred)


def mse(y_test, y_pred):
    """
    Calculates mean squared error between y_test and y_preds.
    """
    return tf.metrics.mean_squared_error(y_test, y_pred)


def run():
    """01.3 Improving model"""

    '''Evaluating predictions'''
    """
    Two of the main metrics used for regression problems are:
    - Mean absolute error (MAE) - the mean difference between each of the predictions.
    - Mean squared error (MSE) - the squared mean difference of the predictions (use if larger errors are more 
    detrimental than smaller errors).
    """

    # Initialize class instance
    data = RegressionData()
    data.train_test_init()

    # Getting a data
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test

    # Set random seed
    # tf.random.set_seed(42)

    # Create a model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1, input_shape=[1])  # define the input_shape to our model
    # ])

    # Compile model
    # model.compile(loss=tf.keras.losses.mae,
    #               optimizer=tf.keras.optimizers.SGD(),
    #               metrics=["mae"])

    # Fit the model to the training data
    # model.fit(X_train, y_train, epochs=100, verbose=0)  # verbose controls how many gets output

    # Make predictions
    # y_preds = model.predict(X_test)

    # Evaluate the model on the test set
    # print(model.evaluate(X_test, y_test))

    # Calculate the mean absolute error
    # mae_var = tf.metrics.mean_absolute_error(y_true=y_test,
    #                                          y_pred=y_preds)
    # print(f"MAE:{mae_var}")
    # Check the test label tensor values
    # print(f"y_test:{y_test}")
    # Check the prediction tensor values (notice the extra square brackets)
    # print(f"y_preds:{y_preds}")
    # Check the tensor shapes
    # print(f"y_test(shape):{y_test.shape}\ny_preds(shape):{y_preds.shape}")

    # Shape before squeeze()
    # print(y_preds.shape)
    # Shape after squeeze()
    # print(y_preds.squeeze().shape)

    # What do they look like?
    # print(f"test: {y_test}\npred(squeezed): {y_preds.squeeze()}")

    # Calculate the MAE
    # mae = tf.metrics.mean_absolute_error(y_true=y_test,
    #                                      y_pred=y_preds.squeeze())  # use squeeze() to make same shape
    # print(f"MAE: {mae}")
    # Calculate the MSE
    # mse = tf.metrics.mean_squared_error(y_true=y_test,
    #                                     y_pred=y_preds.squeeze())
    # print(f"MSE: {mse}")

    # Returns the same as tf.metrics.mean_absolute_error()
    # print(f"MAE (wth TF func): {tf.reduce_mean(tf.abs(y_test - y_preds.squeeze()))}")

    '''Running experiments to improve a model'''
    """
    Again, there are many different ways we can improve it, but 3 of the main ones are:
    - Get more data - get more examples for your model to train on (more opportunities to learn patterns).
    - Make your model larger (use a more complex model) - this might come in the form of more layers or 
    more hidden units in each layer.
    - Train for longer - give model more of a chance to find the patterns in the data.
    
    We build 3 models and compare their results:
    """

    # 1- Build model_1

    # Set random seed
    tf.random.set_seed(42)

    # Replicate original model
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model_1.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['mae'])

    # Fit the model
    model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

    # Make and plot predictions for model_1
    y_preds_1 = model_1.predict(X_test)
    # plot_predictions(X_train, y_train, X_test, y_test, y_preds_1)

    # Calculate model_1 metrics
    mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
    mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
    # print(f"MAE_1:{mae_1}\nMSE_1:{mse_1}")

    # 2- Build model_2

    # Set random seed
    tf.random.set_seed(42)

    # Replicate model_1 and add an extra layer
    model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)  # add a second layer
    ])

    # Compile the model
    model_2.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['mae'])

    # Fit the model
    model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)  # set verbose to 0 for less output

    # Make and plot predictions for model_2
    y_preds_2 = model_2.predict(X_test)
    # plot_predictions(X_train, y_train, X_test, y_test, y_preds_2)

    # Calculate model_2 metrics
    mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
    mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()
    # print(f"MAE_2:{mae_2}\nMSE_2:{mse_2}")

    # 3- Build model_3

    # Set random seed
    tf.random.set_seed(42)

    # Replicate model_2
    model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model_3.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['mae'])

    # Fit the model (this time for 500 epochs, not 100)
    model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0)  # set verbose to 0 for less output

    # Make and plot predictions for model_3
    y_preds_3 = model_3.predict(X_test)
    # plot_predictions(X_train, y_train, X_test, y_test, y_preds_3)

    # Calculate model_3 metrics
    mae_3 = mae(y_test, y_preds_3.squeeze()).numpy()
    mse_3 = mse(y_test, y_preds_3.squeeze()).numpy()
    # print(f"MAE_3:{mae_3}\nMSE_3:{mse_3}")

    '''Comparing results'''

    # Create list in list with results
    model_results = [["model_1", mae_1, mse_1],
                     ["model_2", mae_2, mse_2],
                     ["model_3", mae_3, mse_3]]

    # Convert lists to pandas DataFrame
    final_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
    print(final_results)
