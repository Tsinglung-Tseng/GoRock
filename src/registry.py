import tensorflow as tf

# from .utils.logging_methods import Print


class Registry:
    bi_mapping_items = [
        tf.keras.losses.SparseCategoricalCrossentropy,
        tf.keras.optimizers.Adam,
        tf.keras.metrics.Mean,
        tf.keras.metrics.SparseCategoricalAccuracy,
        tf.keras.layers.Dense,
        tf.keras.layers.Flatten,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Softmax,
        # Print
    ]


class ConfigType:
    DATASET = 'dataset'
    MODEL = 'model'
    SESSION = 'tf_session'

