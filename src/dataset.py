import tensorflow as tf
import numpy as np


class MNISTLoader:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (
            (self.x_train, self.y_train),
            (self.x_test, self.y_test),
        ) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Add a channels dimension
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]

        self.train_ds = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            .shuffle(10000)
            .batch(32)
        )
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        ).batch(32)

    def train_data(self):
        return self.train_ds  # .as_numpy_iterator()

    def test_data(self):
        return self.test_ds  # .as_numpy_iterator()


# class MLP(tf.keras.Model):
# def __init__(self):
# super().__init__()
# self.flatten = tf.keras.layers.Flatten()
# self.dense1 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
# self.dense2 = tf.keras.layers.Dense(units=10)

# def call(self, inputs):         # [batch_size, 28, 28, 1]
# x = self.flatten(inputs)    # [batch_size, 784]
# x = self.dense1(x)          # [batch_size, 100]
# x = self.dense2(x)          # [batch_size, 10]
# output = tf.nn.softmax(x)
# return output
