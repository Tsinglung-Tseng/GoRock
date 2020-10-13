import hashlib
import json


class Hasher:
    @staticmethod
    def raw_config_hash(c):
        return hashlib.sha256(json.dumps(c).encode("utf-8")).hexdigest()

    @staticmethod
    def trainer_hasher(trainer):
        dataset_hash = Hasher.raw_config_hash(trainer.dataset.dump_config())
        model_hash = Hasher.raw_config_hash(trainer.model.dump_config())
        trainer_hash = Hasher.raw_config_hash(trainer.dump_config())
        hashes = [dataset_hash, model_hash, trainer_hash]
        hashes.sort()
        return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
