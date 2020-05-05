


class Session:
    def __init__(self, trainer):
        self.trainer = trainer
        self.variables = None

    def pre_flight_check(self):
        if is_seen_trainer(self.trainer):
            self.trainer = previous_trainer
            self.variables = load_variables(self.trainer)




