import tensorflow as tf


def run():
    """00.1 Getting information from tensors (shape, rank, size)"""
    '''
    Tensor Vocabulary.
    - Shape: The length (number of elements) of each of the dimensions of a tensor.
    - Rank: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1,
    a matrix is rank 2, a tensor has rank n.
    - Axis or Dimension: A particular dimension of a tensor.
    - Size: The total number of items in the tensor.
    '''

    # Create a rank 4 tensor (4 dimensions)
    # rank_4_tensor = tf.zeros([2, 3, 4, 5])
    # print(rank_4_tensor)

    # Get info from tensor
    # print(rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor))

    # Get various attributes of tensor
    # print("Datatype of every element:", rank_4_tensor.dtype)
    # print("Number of dimensions (rank):", rank_4_tensor.ndim)
    # print("Shape of tensor:", rank_4_tensor.shape)
    # print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
    # print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
    # print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy())  # .numpy() converts to NumPy array

    # Get the first 2 items of each dimension
    # print(rank_4_tensor[:2, :2, :2, :2])

    # Get the dimension from each index except for the final one
    # print(rank_4_tensor[:1, :1, :1, :])

    # Create a rank 2 tensor (2 dimensions)
    rank_2_tensor = tf.constant([[10, 7],
                                 [3, 4]])
    # Get the last item of each row
    # print(rank_2_tensor[:, -1])

    # Add an extra dimension (to the end)
    rank_3_tensor = rank_2_tensor[..., tf.newaxis]  # in Python "..." means "all dimensions prior to"
    print(rank_2_tensor, rank_3_tensor)  # shape (2, 2), shape (2, 2, 1)

    # We can achieve the same using tf.expand_dims()
    rank_31_tensor = tf.expand_dims(rank_2_tensor, axis=-1)  # "-1" means last axis
    print(rank_31_tensor)


