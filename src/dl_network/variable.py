from ..registry import FilePath
from ..pg.sql_templates import SQLRunner


class Variable:
    def __init__(self, trainer):
        self.trainer = trainer
        self.variable_path = "/".join([FilePath.VARIABLE, self.trainer.hash + ".h5"])

    def load(self):
        pass

    def save(self, sess_id):
        self.trainer.model.save_weights(self.variable_path)
        SQLRunner.update_varible_on_session_ends(self.variable_path, sess_id)
