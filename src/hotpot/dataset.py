import abc
import json
import h5py

import numpy as np
import tensorflow as tf

from .dl_network.config import FrozenJSON
from .utils.bi_mapper import ConfigBiMapping


class Dataset:
    @abc.abstractmethod
    def train_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def dump_config(self):
        raise NotImplementedError

    @abc.abstractmethod
    def x_shape(self):
        raise NotImplementedError


class MNISTDataset(Dataset):
    def __init__(self, config):
        self.raw_config = ConfigBiMapping.load(config)
        self.config = FrozenJSON(config)
        self.mnist = tf.keras.datasets.mnist
        (
            (self.x_train, self.y_train),
            (self.x_test, self.y_test),
        ) = self.mnist.load_data()

        if self.config.normalization:
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

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
        return self.train_ds

    def test_data(self):
        return self.test_ds

    def dump_config(self):
        return self.raw_config

    @property
    def x_shape(self):
        return (self.config.batch_size, *self.x_train.shape[1:])


class PhantomDataset(Dataset):
    def __init__(self, config):
        self.raw_config = ConfigBiMapping.load(config)
        self.config = FrozenJSON(config)
        self.phantom = h5py.File(self.config.path, "r")["phantom"]

        self.num_dataset_samples = self.phantom.shape[0]

        self.y_image_size = self.phantom.shape[1:]
        self.x_image_size = tuple(
            i // self.config.downsampling_ratio for i in self.y_image_size
        )

        self.num_test = int(self.num_dataset_samples * self.config.test_portion)
        self.num_train = self.num_dataset_samples - self.num_test

        self.y_train, self.y_test = (
            self.phantom[: self.num_train][..., tf.newaxis],
            self.phantom[self.num_train :][..., tf.newaxis],
        )

        self.train_ds = tf.data.Dataset.from_generator(
            self._generator_train,
            output_types=(tf.dtypes.float32, tf.dtypes.float32),
            output_shapes=((*self.x_image_size, 1), (*self.y_image_size, 1)),
        ).batch(self.config.batch_size)

        self.test_ds = tf.data.Dataset.from_generator(
            self._generator_test,
            output_types=(tf.dtypes.float32, tf.dtypes.float32),
            output_shapes=((*self.x_image_size, 1), (*self.y_image_size, 1)),
        ).batch(self.config.batch_size)

    def down_sampler(self, img):
        return tf.image.resize(img, self.x_image_size)

    def _generator_train(self):
        for sample_idx in range(self.num_train):
            yield self.down_sampler(self.y_train)[sample_idx], self.y_train[sample_idx]

    def _generator_test(self):
        for sample_idx in range(self.num_test):
            yield self.down_sampler(self.y_test)[sample_idx], self.y_test[sample_idx]

    def train_data(self):
        return self.train_ds

    def test_data(self):
        return self.test_ds

    @property
    def x_shape(self):
        return (self.config.batch_size, *self.x_image_size, 1)

    def dump_config(self):
        return self.raw_config
