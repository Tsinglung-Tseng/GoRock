import abc
import json

from ..dl_network.trainer import Trainer
from ..pg.pool import server_side_cursor
from ..pg.sql_templates import SQLRunner, Template
from ..registry import ConfigType
from .bi_mapper import ConfigBiMapping


class Logger:
    @abc.abstractmethod
    def __call__(self):
        pass


# class Print(Logger):
# def __init__(self, trainer: Trainer):
# pass

# def __call__(self):
# return print


class LogAndPrint(Logger):
    def __init__(self, trainer: Trainer):
        self.dataset_config = trainer.dataset.dump_config()
        self.model_config = trainer.model.dump_config()
        self.trainer_config = trainer.dump_config()

        self.prior_session_id = None
        self.session_reference = {
            "dataset_config_id": SQLRunner.insert_config_if_not_exist(
                ConfigType.DATASET, self.dataset_config
            ),
            "model_config_id": SQLRunner.insert_config_if_not_exist(
                ConfigType.MODEL, self.model_config
            ),
            "trainer_config_id": SQLRunner.insert_config_if_not_exist(
                ConfigType.TRAINER, self.trainer_config
            ),
        }
        self.session_id = SQLRunner.create_or_add_cascade_sessoion(self)
        self.session_process = {"session_id": self.session_id, "epoch": 0}

    def log_epoch_progress(self, trainer):
        self.session_process.update(
            {
                "epoch": self.session_process["epoch"] + 1,
                "loss": float(trainer.train_loss.result()),
                "accuracy": float(trainer.train_accuracy.result() * 100),
                "test_error": float(trainer.test_loss.result()),
                "test_accuracy": float(trainer.test_accuracy.result() * 100),
            }
        )
        SQLRunner.insert_sessoion_log(self)
        print(self.session_process)

    def on_session_start(self):
        SQLRunner.update_session_on_start(self)

    def on_session_end(self):
        SQLRunner.update_session_on_end(self)
