from hotpot.simulation.image_system import ImageSystem
from hotpot.geometry.primiary import Cartesian3, Surface, Box
from hotpot.sample import FuncArray
from hotpot.geometry.system import SipmArray
import plotly.graph_objects as go
import tensorflow as tf


class Sample:
    def __init__(self, sample_df):
        self.sample_df = sample_df

    @staticmethod
    def from_database(experiment_ids):
        sample_stmt = """
        SELECT
            ts.*
        FROM
            train_sample ts
            JOIN experiment_coincidence_event ece ON (ts. "eventID" = ece. "eventID")
            JOIN experiments e ON (ece.experiment_id = e.id)
        WHERE
            ts.gamma_1_x IS NOT NULL
            AND experiment_id = 8;
        """
        
        sample_df = Database().read_sql(sample_stmt)
        return Sample(sample_df)
        
    @property
    def counts(self):
        return FuncArray.from_pd_series(self.sample_df.counts)
    
    @property
    def sipm_center_pos(self):
        return FuncArray.from_pd_series(self.sample_df.sipm_center_pos)
    
    @property
    def train_sample(self):
        return self.counts.expand_dims(2).concatenate_with(self.sipm_center_pos, 2).shrink((1,2)).rollaxis(1,4).to_tensor()
    
    @property
    def train_label(self):
        return FuncArray(self.sample_df[['gamma_1_x', 'gamma_1_y','gamma_1_z','gamma_2_x','gamma_2_y','gamma_2_z']]).to_tensor()
    
    @property
    def train_ds(self):
        return tf.data.Dataset.from_tensor_slices((self.train_sample, self.train_label))

