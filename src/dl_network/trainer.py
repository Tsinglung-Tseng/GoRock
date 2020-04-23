import tensorflow as tf
from .config import FrozenJSON


class Trainer:
    def __init__(self, dataset, model: tf.Module, config):
        self.dataset = dataset

        self.model = model
        self.config = FrozenJSON(config)

        self.epoch = self.config.epoch

        self.loss_object = self.config.loss_object
        self.optimizer = self.config.optimizer
        self.train_loss = self.config.train_loss
        self.train_accuracy = self.config.train_accuracy

        self.test_loss = self.config.test_loss
        self.test_accuracy = self.config.test_accuracy

    def _log_config(self):
        if self.logger is None:
            pass
        else:
            self.logger.log_network_config(self.config)
            self.logger.log_
            #TODO

    def run(self):
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

            template = """Epoch: {}, Loss: {}, Accuracy: {}, Test Loss:{}, Test Accuracy: {}."""
            print(
                template.format(
                    epoch + 1,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.test_loss.result(),
                    self.test_accuracy.result() * 100,
                )
            )
