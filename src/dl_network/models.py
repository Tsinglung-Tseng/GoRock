from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from ..utils.bi_mapper import ConfigBiMapping


class Conv2DClassification(Model):
    def __init__(self):
        super(Conv2DClassification, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.softmax(x)


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


class GeneralModel(Model):
    def __init__(self, config):
        super(GeneralModel, self).__init__()
        self.config = config
        self._sub_models = []
        for sub_model in self._sub_models:
            sub_model_type = list
