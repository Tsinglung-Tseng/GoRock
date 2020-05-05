import abc
from ..pg.pool import server_side_cursor
from ..dl_network.trainer import Trainer
from ..pg.sql_templates import Template, SQLRunner
from ..registry import ConfigType
import json
from .bi_mapper import ConfigBiMapping


class LoggerMethod:
    @abc.abstractmethod
    def __call__(self):
        pass


class Print(LoggerMethod):
    def __init__(self, sess: Trainer):
        pass

    def __call__(self):
        return print


class LogAndPrint(LoggerMethod):
    def __init__(self, sess: Trainer):
        self.dataset_config = sess.dataset.dump_config()
        self.model_config = sess.model.dump_config()
        self.sess_config = sess.dump_config()

        self.dataset_config_id = SQLRunner.insert_config_if_not_exist(
            ConfigType.DATASET, self.dataset_config
        )
        self.model_config_id = SQLRunner.insert_config_if_not_exist(
            ConfigType.MODEL, self.model_config
        )
        self.tf_session_config_id = SQLRunner.insert_config_if_not_exist(
            ConfigType.SESSION, self.sess_config
        )
        self.data_to_log = {
            'dataset_config_id': self.dataset_config_id,
            'model_config_id': self.model_config_id,
            'tf_session_config_id': self.tf_session_config_id,
        }

    def __call__(self):
        def _log_and_print(sess_progress):
            self.data_to_log.update(sess_progress)
            SQLRunner.insert_sessoion_log(self.data_to_log)
            print(self.data_to_log)
        return _log_and_print

