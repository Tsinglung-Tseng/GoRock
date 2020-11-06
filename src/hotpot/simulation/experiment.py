from .mac import MAC
from ..database import Database
import pandas as pd


class Experiment:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
    
    @property
    def experiment_id(self):
        return self.experiment_config['experiment_id'][0]
    
    @property
    def path(self):
        return self.experiment_config['path'][0]
    
    @property
    def coincidence_count(self):
        return self.experiment_config['coincidence_count'][0]
    
    @property
    def geometry_mac(self):
        return MAC.from_database(self.experiment_config['geometry_mac_id'][0])
    
    @property
    def source_mac(self):
        return MAC.from_database(self.experiment_config['source_mac_id'][0])
    
    @staticmethod
    def from_database(experiment_id):
        experiment_config = pd.read_sql(f"""
            SELECT
                *
            FROM
                experiment_config
            WHERE
                experiment_id = {experiment_id};
            """, con=Database().engine())
        return Experiment(experiment_config)
