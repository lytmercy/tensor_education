import tensorflow as tf
import numpy as np


def run():
    """00.2 Manipulating tensors"""

    '''Basic operations'''
    # We can perform many of the basic mathematical operations directly on tensors (+, -, *)
    tensor = tf.constant([[10, 7], [3, 4]])
    # print(tensor)
    # Add values to a tensor using the addition operator
    # print(tensor + 10)
    # Multiplication (known as element-wise multiplication)
    # print(tensor * 10)
    # Subtraction
    # print(tensor - 10)

    # Use the tensorflow function equivalent of the '*' (multiply) operator
    # print(tf.multiply(tensor, 10))

    '''Matrix multiplication'''
    """
    Two rules for matrix multiplication to remember are:
    1. The inner dimensions must match:
    - (3, 5) @ (3, 5) won't work
    - (5, 3) @ (3, 5) will work
    - (3, 5) @ (5, 3) will work
    2. The resulting matrix has the shape of the outer dimensions:
    - (5, 3) @ (3, 5) -> (5, 5)
    - (3, 5) @ (5, 3) -> (3, 3)
    """

    # Matrix multiplication in TensorFlow
    # print(tensor)
    # print(tf.matmul(tensor, tensor))
    # Matrix multiplication with Python operator '@'
    # print(tensor @ tensor)

    # Create (3, 2) tensor
    X = tf.constant([[1, 2],
                     [3, 4],
                     [5, 6]])
    # Create (3, 2) tensor
    Y = tf.constant([[7, 8],
                     [9, 10],
                     [11, 12]])
    # print(X, Y)
    # Try to matrix multiply them (will error)
    # print(X @ Y)
    # Errors because the inner dimensions don't match.
    """
    We need to either:
    - Reshape X to (2, 3) so it's (2, 3) @ (3, 2)
    - Reshape Y to (3, 2) so it's (3, 2) @ (2, 3)
    
    We can do this with either:
    - tf.reshape() - allows us to reshape a tensor into a defined shape.
    - tf.transpose() - switches the dimensions of a given tensor.
    """

    # Example of reshape (3, 2) -> (2, 3)
    # Y_reshaped = tf.reshape(Y, shape=(2, 3))
    # print(Y_reshaped)
    # Try matrix multiplication with reshaped Y
    # print(X @ Y_reshaped)

    # Example of transpose (3, 2) -> (2, 3)
    # X_transposed = tf.transpose(X)
    # Try matrix multiplication
    # print(tf.matmul(X_transposed, Y))

    # We can achieve the same result with parameters
    # print(tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False))

    '''The dot product'''
    # Perform the dot product on X and Y (requires X to be transposed)
    # print(tf.tensordot(tf.transpose(X), Y, axes=1))

    # reshape and transpose work, but we get different results when using each.
    # Perform matrix multiplication between X and Y (transposed)
    # print(tf.matmul(X, tf.transpose(Y)))

    # Perform matrix multiplication between X and Y (reshaped)
    # print(tf.matmul(X, tf.reshape(Y, (2, 3))))

    # Check shapes of Y, reshaped Y and transposed Y
    # print("Y shape:", Y.shape)
    # print("Y (reshaped) shape:", tf.reshape(Y, (2, 3)).shape)
    # print("Y (transpose) shape:", tf.transpose(Y).shape)
    # print("\n")

    # Check values of Y, reshaped Y and transposed Y
    # print("Normal Y:")
    # print(Y, "\n")
    #
    # print("Y reshaped to (2, 3):")
    # print(tf.reshape(Y, (2, 3)), "\n")
    #
    # print("Y transposed:")
    # print(tf.transpose(Y))

    '''Changing the datatype of a tensor'''



