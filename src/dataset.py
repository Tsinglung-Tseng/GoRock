import tensorflow as tf
import numpy as np
from .dl_network.config import FrozenJSON


class MNISTLoader:
    def __init__(self, config):
        self.config = FrozenJSON(config)
        self.mnist = tf.keras.datasets.mnist
        (
            (self.x_train, self.y_train),
            (self.x_test, self.y_test),
        ) = self.mnist.load_data()
    
        if self.config.normalization:
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Add a channels dimension
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]

        self.train_ds = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            .shuffle(self.config.shuffle)
            .batch(self.config.batch_size)
        )
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        ).batch(self.config.batch_size)

    def train_data(self):
        return self.train_ds  # .as_numpy_iterator()

    def test_data(self):
        return self.test_ds  # .as_numpy_iterator()

