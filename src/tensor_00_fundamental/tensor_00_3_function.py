import tensorflow as tf
import numpy as np


def run():
    """00.3 Using @tf.function"""

    # Create a simple function
    def function(x, y):
        return x ** 2 + y

    x = tf.constant(np.arange(0, 10))
    y = tf.constant(np.arange(10, 20))
    # print(function(x, y))

    # Create the same function and decorate it with tf.function
    @tf.function
    def tf_function(x, y):
        return x ** 2 + y

    print(tf_function(x, y))


