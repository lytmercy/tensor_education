import tensorflow as tf
import numpy as np


def run():
    """00.2 Manipulating tensors"""

    '''Basic operations'''
    # We can perform many of the basic mathematical operations directly on tensors (+, -, *)
    # tensor = tf.constant([[10, 7], [3, 4]])
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
    # X = tf.constant([[1, 2],
    #                  [3, 4],
    #                  [5, 6]])
    # Create (3, 2) tensor
    # Y = tf.constant([[7, 8],
    #                  [9, 10],
    #                  [11, 12]])
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

    # Create a new tensor with default datatype (float32)
    # B = tf.constant([1.7, 7.4])
    # Create a new tensor with default datatype (int32)
    # C = tf.constant([1, 7])
    # print(B, C)

    # Change from float32 to float16 (reduced precision)
    # B = tf.cast(B, dtype=tf.float16)
    # print(B)

    # Change from int32 to float32
    # C = tf.cast(C, dtype=tf.float32)
    # print(C)

    '''Getting absolute value'''
    # Create tensor with negative values
    # D = tf.constant([-7, -10])
    # print(D)

    # Get the absolute values
    # print(tf.abs(D))

    '''Finding the min, max, mean, sum (aggregation)'''

    # Create a tensor with 50 random values between 0 and 100
    # E = tf.constant(np.random.randint(low=0, high=100, size=50))
    # print(E)
    # Find the minimum
    # print("Min(E): ", tf.reduce_min(E))
    # Find the maximum
    # print("Max(E): ", tf.reduce_max(E))
    # Find the mean
    # print("Mean(E): ", tf.reduce_mean(E))
    # Find the sum
    # print("Sum(E): ", tf.reduce_sum(E))

    '''Finding the positional maximum and minimum'''

    # Create a tensor with 50 values between 0 and 1
    # F = tf.constant(np.random.random(50))
    # print(F)
    # Find the maximum element position of F
    # print(tf.argmax(F))
    # Find the minimum element position of F
    # print(tf.argmin(F))

    # Find the maximum element position of F
    # print(f"The maximum value of F is at position: {tf.argmax(F).numpy()}")
    # print(f"The maximum value of F is: {tf.reduce_max(F).numpy()}")
    # print(f"Using tf.argmax() to index F, the maximum value of F is: {F[tf.argmax(F)].numpy()}")
    # print(f"Are the two max values the same (they should be)? - {F[tf.argmax(F)].numpy() == tf.reduce_max(F).numpy()}")

    '''Squeezing a tensor (removing all single dimensions'''

    # Create a rank 5 (5 dimensions) tensor of 50 numbers between 0 and 100
    # G = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
    # print(f"Shape: {G.shape}")
    # print(f"Dimensions: {G.ndim}")
    # print()

    # Squeeze tensor G (remove all 1 dimensions
    # G_squeezed = tf.squeeze(G)
    # print(f"Shape: {G_squeezed.shape}")
    # print(f"Dimensions: {G_squeezed.ndim}")

    '''One-hot encoding'''

    # Create a list of indices
    # some_list = [0, 1, 2, 3]

    # One-hot encode them
    # print(tf.one_hot(some_list, depth=4))

    # Specify custom values for 'on' and 'off' encoding
    # print(tf.one_hot(some_list, depth=4, on_value="We're live!", off_value="Offline"))

    '''Squaring, log, square root'''
    """
    tf.sqrt() - get the square root of every value in a tensor (note: the elements need to be floats or this will error).
    tf.math.log() - get the natural log of every value in a tensor (elements need to floats).
    """

    # Create a new tensor
    H = tf.constant(np.arange(1, 10))
    # print(H)

    # Square it
    # print(tf.square(H))

    # Find the square root (will error), needs to be non-integer
    # print(tf.sqrt(H))

    # Change data type of tensor to float32
    # H = tf.cast(H, dtype=tf.float32)
    # print(tf.sqrt(H))

    # Find the log (input also needs to be float)
    # print(tf.math.log(H))

    '''Manipulating tf.Variable tensors'''

    # Create a variable tensor
    # I = tf.Variable(np.arange(0, 5))
    # print(I)

    # Assign the final value a new value of 50
    # print(I.assign([0, 1, 2, 3, 50]))

    # Add 10 to every element in I
    # print(I.assign_add([10, 10, 10, 10, 10]))

    '''Tensors and Numpy'''

    # Create a tensor from a Numpy array
    J = tf.constant(np.array([3., 7., 10.]))
    # print(J)

    # Convert tensor J to NumPy with np.array()
    # print(np.array(J), type(np.array(J)))

    # Convert tensor J to NumPy with .numpy()
    # print(J.numpy(), type(J.numpy()))

    # Create a tensor from NumPy and from an array
    numpy_J = tf.constant(np.array([3., 7., 10.]))  # will be float64 (due to NumPy)
    tensor_J = tf.constant([3., 7., 10.])  # will be float32 (due to being TensorFlow default)

    print(numpy_J.dtype, tensor_J.dtype)
