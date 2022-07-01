import tensorflow as tf
import numpy as np


def run():
    """00.4 Exercises"""
    """
    1. Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
    2. Find the shape, rank and size of the tensors you created in 1.
    3. Create two tensors containing random values between 0 and 1 with shape [5, 300].
    4. Multiply the two tensors you created in 3 using matrix multiplication.
    5. Multiply the two tensors you created in 3 using dot product.
    6. Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
    7. Find the min and max values of the tensor you created in 6.
    8. Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3].
    9. Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
    10. One-hot encode the tensor you created in 9.
    """

    '''Exercise - 1'''

    # Create a scalar
    # scalar = tf.constant(384)
    # Create a vector
    # vector = tf.constant([38., 20., 4., 18.])
    # Create a matrix
    # matrix = tf.constant([[32, 324, 98, 199],
    #                       [23, 723, 83, 27]])
    # Create a tensor
    # tensor = tf.constant([[[328, 324, 389],
    #                        [349, 139, 389]],
    #                       [[393, 329, 193],
    #                        [499, 758, 738]]])

    # Print all
    # print(f"Scalar: {scalar}")
    # print(f"Vector: {vector}")
    # print(f"Matrix:\n{matrix}")
    # print(f"Tensor:\n{tensor}")

    '''Exercise - 2'''

    # Print shape, rank, and size of all above-created tensors
    # print(f"Scalar. shape={scalar.shape}, rank={scalar.ndim}, size={tf.size(scalar)}")
    # print(f"Vector. shape={vector.shape}, rank={vector.ndim}, size={tf.size(vector)}")
    # print(f"Matrix. shape={matrix.shape}, rank={matrix.ndim}, size={tf.size(matrix)}")
    # print(f"Tensor. shape={tensor.shape}, rank={tensor.ndim}, size={tf.size(tensor)}")

    '''Exercise - 3'''

    # Creating two tensors with random values
    # gen = tf.random.Generator.from_non_deterministic_state()
    # T = gen.uniform(shape=[5, 300])
    # F = gen.uniform(shape=[5, 300])
    # print(f"Tensor T:\n{T}")
    # print(f"Tensor F:\n{F}")

    '''Exercise - 4'''

    # Matrix multiplication
    # print(tf.matmul(T, tf.transpose(F)))

    '''Exercise - 5'''

    # Matrix dot product
    # print(tf.tensordot(tf.transpose(T), F, axes=1))

    '''Exercise - 6'''

    # Create a tensor with random values and shape [224, 224, 3]
    # tensor_3d = gen.uniform(shape=[224, 224, 3])

    '''Exercise - 7'''

    # Find min and max values
    # print(f"Min value: {tf.reduce_min(tensor_3d).numpy()}")
    # print(f"Max value: {tf.reduce_max(tensor_3d).numpy()}")

    '''Exercise - 8'''

    # Create a random tensor with shape [1, 224, 224, 3]
    # tensor_before_squeeze = gen.normal(shape=[1, 224, 224, 3])
    # print(f"Shape: {tensor_before_squeeze.shape}")

    # Squeeze tensor
    # tensor_after_squeeze = tf.squeeze(tensor_before_squeeze)
    # print(f"Shape: {tensor_after_squeeze.shape}")

    '''Exercise - 9'''

    # Create a tensor with shape = [10]
    vec_tensor = tf.random.Generator.from_seed(5).uniform(shape=[10], dtype=tf.int32, maxval=20)
    print(vec_tensor)

    # Find index of max value
    print(f"Max index: {tf.argmax(vec_tensor).numpy()}, value: {tf.reduce_max(vec_tensor).numpy()}")

    '''Exercise - 10'''

    # Perform one hot encoding on tensor from exercise 9
    print(tf.one_hot(vec_tensor, depth=10))
