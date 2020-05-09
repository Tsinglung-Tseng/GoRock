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
