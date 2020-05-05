import abc
from ..pg.pool import server_side_cursor
from ..dl_network.trainer import Trainer
from ..pg.sql_templates import Template, SQLRunner
from ..registry import ConfigType
import json
from .bi_mapper import ConfigBiMapping


class Logger:
    @abc.abstractmethod
    def __call__(self):
        pass


class Print(Logger):
    def __init__(self, trainer: Trainer):
        pass

    def __call__(self):
        return print


class LogAndPrint(Logger):
    def __init__(self, trainer: Trainer):
        self.dataset_config = trainer.dataset.dump_config()
        self.model_config = trainer.model.dump_config()
        self.trainer_config = trainer.dump_config()

        self.prior_session_id = None
        self.session_reference = {
            "dataset_config_id": 
                SQLRunner.insert_config_if_not_exist(
                    ConfigType.DATASET, self.dataset_config
                ),
            "model_config_id": 
                SQLRunner.insert_config_if_not_exist(
                    ConfigType.MODEL, self.model_config
                ),
            "trainer_config_id":
                SQLRunner.insert_config_if_not_exist(
                    ConfigType.TRAINER, self.trainer_config
                )
        }
        self.session_id = SQLRunner.create_or_add_cascade_sessoion(self) 
        self.session_process = {'session_id': self.session_id} 

    def __call__(self):
        def _log_and_print(sess_progress):
            self.session_process.update(sess_progress)
            SQLRunner.insert_sessoion_log(self)
            print(self.session_process)

        return _log_and_print
