import json

import tensorflow as tf

from ..utils.bi_mapper import ConfigBiMapping
from .config import FrozenJSON
from ..dl_network.hasher import Hasher
from .variable import Variable


class Trainer:
    def __init__(self, dataset, model: tf.Module, config, logger, variables=None):
        self.dataset = dataset

        self.model = model
        self.raw_config = config
        self.config = ConfigBiMapping.load(self.raw_config)
        self.config = FrozenJSON(self.config)

        self.epoch = self.config.epoch

        self.loss_object = self.config.loss_object
        self.optimizer = self.config.optimizer
        self.train_loss = self.config.train_loss
        self.train_accuracy = self.config.train_accuracy

        self.test_loss = self.config.test_loss
        self.test_accuracy = self.config.test_accuracy

        self.logger = logger(self)

        self.variables = variables

    def dump_config(self):
        return self.raw_config 
        # return ConfigBiMapping.dump(self.raw_config)

    @property
    def hash(self):
        return Hasher.trainer_hasher(self)

    def build(self):
        pass
        # TODO: Build model from existing variable

    def run(self):
        self.build()
        self.logger.on_session_start()

        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs)
                loss = self.loss_object(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        @tf.function
        def test_step(inputs, labels):
            predictions = self.model(inputs)
            t_loss = self.loss_object(labels, predictions)

            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)

        for epoch in range(self.epoch):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for inputs, labels in self.dataset.train_data():
                train_step(inputs, labels)

            for test_inputs, test_labels in self.dataset.test_data():
                test_step(test_inputs, test_labels)

            self.logger.log_epoch_progress(self)

        self.variable = Variable(self)
        self.variable.save(self.logger.session_id)
        physical_devices.logger.on_session_end()
