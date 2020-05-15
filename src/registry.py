import tensorflow as tf
from functools import partial


class TFMethodWrapper:
    @staticmethod
    def MSE():
        return tf.keras.losses.MSE

    @staticmethod
    def Adam_SRCNN_1eNeg5():
        return partial(tf.keras.optimizers.Adam, learning_rate=0.00001)()


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
        TFMethodWrapper.MSE,
        TFMethodWrapper.Adam_SRCNN_1eNeg5,
        tf.keras.metrics.MeanSquaredError,
        tf.keras.layers.UpSampling2D
    ]


class ConfigType:
    DATASET = "dataset"
    MODEL = "model"
    TRAINER = "trainer"
    # SESSION = "tf_session"


class TableName:
    SESSIONLOG = "session_log"
    SESSION = "session"


class FilePath:
    VARIABLE = "/mnt/users/qinglong/variables"


