import tensorflow as tf


def fix_tf_gpu():

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass