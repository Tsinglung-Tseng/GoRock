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
        self.sess_config_id = SQLRunner.insert_config_if_not_exist(
            ConfigType.SESSION, self.sess_config
        )

    def __call__(self):
        def _log_and_print(data_to_log):
            # TODO: bind log to session
            with server_side_cursor() as cur:
                cur.execute()
                result = cur.fetchall()

        return _log_and_print

