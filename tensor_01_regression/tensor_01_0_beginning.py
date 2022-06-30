import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def run():
    """01.0 Beginning in regression"""

    '''Creating data to view and fit'''
    # Create features
    X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

    # Create labels
    y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

    # Visualize it
    plt.scatter(X, y)
    # plt.show()

    '''Regression input shapes and output shapes'''
    # Example input and output shapes of a regresson model
    house_info = tf.constant(["bedroom", "bathroom", "garage"])
    house_price = tf.constant([939700])
    # print(house_info)
    # print(house_price)
    # print(house_info.shape)

    # Take a single example of X
    input_shape = X[0].shape

    # Take a single example of y
    output_shape = y[0].shape

    # print(input_shape)
    # print(output_shape)  # these are both scalars (no shape)

    # Let's take a look at the single examples individually
    print(f"X:", X[0])
    print(f"y:", y[0])



