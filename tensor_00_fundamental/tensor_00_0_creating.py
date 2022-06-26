import tensorflow as tf
import numpy as np


def run():
    """00. Getting started with TensorFlow"""

    '''Creating Tensors with tf.constant()'''
    # Create a scalar (rank 0 tensor)
    # scalar = tf.constant(7)
    # print(scalar)

    # Check the number of dimensions of a tensor (ndim stands for number of dimensions)
    # print(scalar.ndim)

    # Create a vector (more than 0 dimensions)
    # vector = tf.constant([10, 10])
    # print(vector)

    # Check the number of dimensions of our vector tensor
    # print(vector.ndim)

    # Create a matrix (more than 1 dimension)
    # matrix = tf.constant([[10, 7],
    #                      [7, 10]])
    # print(matrix)

    # Check the number of dimensions of our matrix tensor
    # print(matrix.ndim)

    # Create another matrix and define the datatype
    # another_matrix = tf.constant([[10., 8.],
    #                               [3., 2.],
    #                               [8., 9.]], dtype=tf.float16)  # specify the datatype with 'dtype'
    # print(another_matrix)

    # Even though another_matrix contains more numbers, its dimensions stay the same
    # print(another_matrix.ndim)

    # Create tensor (more than 2 dimensions)
    # tensor = tf.constant([[[1, 2, 3],
    #                        [4, 5, 6]],
    #                       [[7, 8, 9],
    #                        [10, 11, 12]],
    #                       [[13, 14, 15],
    #                        [16, 17, 18]]])
    # print(tensor)
    # print(tensor.ndim)

    '''Creating Tensors with tf.Variable()'''
    # Create the same tensor with tf.Variable() and tf.constant()
    # changeable_tensor = tf.Variable([10, 7])
    # unchangeable_tensor = tf.constant([10, 7])
    # print(changeable_tensor, unchangeable_tensor)

    # Will error (requires the .assign() method)
    # changeable_tensor[0] = 7
    # print(changeable_tensor)

    # Won't error
    # changeable_tensor[0].assign(7)
    # print(changeable_tensor)

    # Will error (can't change tf.constant())
    # unchangeable_tensor[0].assign(7)
    # print(unchangeable_tensor)

    '''Creating random tensors'''
    # Create two random (but the same) tensors
    # random_1 = tf.random.Generator.from_seed(42)    # set the seed for reproducibility
    # random_11 = random_1.normal(shape=(3, 2))    # create tensor from a normal distribution
    # random_2 = tf.random.Generator.from_seed(42)
    # random_22 = random_2.normal(shape=(3, 2))

    # Are they equal?
    # print(f"Equality = {random_11 == random_22}")
    # print("====================")
    # print(f"Random object: {random_1}")
    # print(f"Random tensor = {random_11}")
    # print("---------------------")
    # print(f"Random object: {random_2}")
    # print(f"Random tensor = {random_22}")

    # Create two random (and different) tensors
    # random_3 = tf.random.Generator.from_seed(42)
    # random_33 = random_3.normal(shape=(3, 2))
    # random_4 = tf.random.Generator.from_seed(11)
    # random_44 = random_4.normal(shape=(3, 2))

    # Check the tensors and see if they are equal
    # print(f"Equality = {random_11 == random_33}")
    # print(f"Equality = {random_33 == random_44}")
    # print("====================")
    # print(f"Random object: {random_3}")
    # print(f"Random tensor = {random_33}")
    # print("---------------------")
    # print(f"Random object: {random_4}")
    # print(f"Random tensor = {random_44}")

    # Shuffle a tensor (valuable for when you want to shuffle your data)
    # not_shuffled = tf.constant([[10, 7],
    #                             [3, 4],
    #                             [2, 5]])
    # Gets different results each time
    # shuffled_tensor = tf.random.shuffle(not_shuffled)
    # print(shuffled_tensor)

    # Shuffle in the same order every time using the seed parameter (won't actually be the same)
    # shuffled_tensor = tf.random.shuffle(not_shuffled, seed=42)
    # print(shuffled_tensor)

    # Shuffle in the same order every time

    # Set the global random seed
    # tf.random.set_seed(42)

    # Set the operation random seed
    # shuffled_tensor = tf.random.shuffle(not_shuffled, seed=42)
    # print(shuffled_tensor)

    # Set the global random seed
    # tf.random.set_seed(42)  # if comment this out we'll get different results

    # Set the operation random seed
    # shuffled_tensor = tf.random.shuffle(not_shuffled)
    # print(shuffled_tensor)

    '''Other way to make tensors'''
    # Make a tensor of all ones
    # ones = tf.ones(shape=(3, 2))
    # print(ones)

    # Make a tensor of all zeros
    # zeros = tf.zeros(shape=(3, 2))
    # print(zeros)

    # Creating tensor from Numpy array
    numpy_A = np.arange(1, 25, dtype=np.int32)  # create a NumPy array between 1 and 25
    A = tf.constant(numpy_A,
                    shape=[2, 4, 3])  # note: the shape total (2*4*3) has to match the number of elements in the array
    print(numpy_A)
    print(A)




