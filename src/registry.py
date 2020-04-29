import tensorflow as tf


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
    ]


class ConfigType:
    DATASET = 'dataset'
    MODEL = 'model'
    SESSION = 'tf_session'

