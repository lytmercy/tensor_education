import tensorflow as tf


def run():
    print(tf.__version__)   # find version number (should be 2.x+)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
