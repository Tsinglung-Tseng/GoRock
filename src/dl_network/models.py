from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import json
from enum import Enum
from ..utils.bi_mapper import ConfigBiMapping
from ..utils.tf_wrapper_crasher import TFWrapperCrasher


# class Conv2DClassification(Model):
    # def __init__(self):
        # super(Conv2DClassification, self).__init__()
        # self.flatten = Flatten()
        # self.d1 = Dense(128, activation="relu")
        # self.d2 = Dense(10, activation="softmax")
        # self.softmax = tf.keras.layers.Softmax()

    # def call(self, x):
        # x = self.flatten(x)
        # x = self.d1(x)
        # x = self.d2(x)
        # return self.softmax(x)


class SeqModel(Model):
    def __init__(self, config):
        super(SeqModel, self).__init__()
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
        # if len(config) != 1:
            # raise ValueError("TF dose not support concate multiple models, use one only!")
        self.config = config
        
    def __call__(self):
        return Models.load(list(self.config.keys())[0])(list(self.config.values())[0])


