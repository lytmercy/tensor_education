import tensorflow as tf


def run():
    """00. Getting started with TensorFlow"""

    '''Creating Tensors with tf.constant()'''
    # Create a scalar (rank 0 tensor)
    scalar = tf.constant(7)
    # print(scalar)

    # Check the number of dimensions of a tensor (ndim stands for number of dimensions)
    # print(scalar.ndim)

    # Create a vector (more than 0 dimensions)
    vector = tf.constant([10, 10])
    # print(vector)

    # Check the number of dimensions of our vector tensor
    # print(vector.ndim)

    # Create a matrix (more than 1 dimension)
    matrix = tf.constant([[10, 7],
                         [7, 10]])
    # print(matrix)

    # Check the number of dimensions of our matrix tensor
    # print(matrix.ndim)

    # Create another matrix and define the datatype
    another_matrix = tf.constant([[10., 8.],
                                  [3., 2.],
                                  [8., 9.]], dtype=tf.float16)  # specify the datatype with 'dtype'
    # print(another_matrix)

    # Even though another_matrix contains more numbers, its dimensions stay the same
    # print(another_matrix.ndim)

    # Create tensor (more than 2 dimensions)
    tensor = tf.constant([[[1, 2, 3],
                           [4, 5, 6]],
                          [[7, 8, 9],
                           [10, 11, 12]],
                          [[13, 14, 15],
                           [16, 17, 18]]])
    # print(tensor)
    # print(tensor.ndim)

    '''Creating Tensors with tf.Variable()'''
    # Create the

