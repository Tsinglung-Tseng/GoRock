from __future__ import absolute_import, division, print_function, unicode_literals

import json
from enum import Enum

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from ..utils.bi_mapper import ConfigBiMapping
from ..utils.tf_wrapper_crasher import TFWrapperCrasher


class SeqModel(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._layers = []
        for layer in self.config:
            layer_type = list(layer.keys())[0]
            layer_parm = list(layer.values())[0]
            self._layers.append(ConfigBiMapping.load_mapping[layer_type](**layer_parm))

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def dump_config(self):
        return TFWrapperCrasher({self.__class__.__name__: list(self.config)})()


class Models:
    __models = [SeqModel]

    load_mapping = {m.__name__: m for m in __models}

    @staticmethod
    def load(model_name):
        return Models.load_mapping[model_name]


class ModelBuilder:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        return Models.load(list(self.config.keys())[0])(list(self.config.values())[0])


class IncidentModelSeqModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(0.5, input_shape=(32, 16, 16, 8))
        self.conv_2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3)
        self.conv_2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3)
        self.conv_2d_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(128)
        self.dense_2 = tf.keras.layers.Dense(64)
        self.dense_3 = tf.keras.layers.Dense(32)
        self.dense_4 = tf.keras.layers.Dense(16)
        self.dense_5 = tf.keras.layers.Dense(6)

    def call(self, x):
        output = self.dropout(x)
        output = self.conv_2d_1(x)
        output = self.conv_2d_2(output)
        output = self.conv_2d_3(output)

        output = self.flatten(output)

        output = self.dense_1(output)
        output = self.dense_2(output)
        output = self.dense_3(output)
        output = self.dense_4(output)
        output = self.dense_5(output)
        return output
