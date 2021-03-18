from hotpot.simulation.image_system import ImageSystem
from hotpot.geometry.primiary import Cartesian3, Surface, Box
# from hotpot.sample import FuncArray
from ..functools import FuncArray
from hotpot.geometry.system import SipmArray
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
from ..database import Database


class Segment:
    def __init__(self, pair):
        self.pair = pair

    def to_plotly(self):
        x=self.pair.reshape([2,3]).T[0]
        y=self.pair.reshape([2,3]).T[1]
        z=self.pair.reshape([2,3]).T[2]
        return go.Scatter3d(x=x,y=y,z=z)

def rotate_matrix(rv_ypr):
    def rotation_matrix_x(angle):
        return tf.convert_to_tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ], dtype=tf.float64
        )

    def rotation_matrix_y(angle):
        return tf.convert_to_tensor(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0, np.cos(angle)],
            ], dtype=tf.float64
        )

    def rotation_matrix_z(angle):
        return tf.convert_to_tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=tf.float64
        )

    return tf.matmul(
            tf.matmul(
                rotation_matrix_x(rv_ypr[0]),
                rotation_matrix_y(rv_ypr[1])
            ),
            rotation_matrix_z(rv_ypr[2])
        )


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


class SampleWithAnger(Sample):
    INDICES = tf.range(16, dtype=tf.float64)
    ROW_AXIS = 1
    COLUMN_AXIS = 0
    SIPM_GAP = 3.26
    XEDGES = tf.convert_to_tensor(np.linspace(-25,25,17))
    YEDGES = tf.convert_to_tensor(np.linspace(-25,25,17))
    
    def __init__(self, train_sample):
        super().__init__(train_sample)
        self.move_args = None
        if self.move_args is None:
            self.build()
    
    def build(self):
        cma = tf.convert_to_tensor(
            Database().read_sql(
                """select * from pos_local_to_global_view;""")[
                    ['move_x','move_y','move_z','rotate_angle_x','rotate_angle_y','rotate_angle_z']
                ].to_numpy()
        )
        cryID = FuncArray.from_pd_series(self.sample_df['crystalID']).to_tensor()
        self.move_args = tf.gather(cma, cryID)
    
    @property
    def gamma_1_counts(self):
        return tf.transpose(self.train_sample[:,:,:,0], perm=[0,2,1])[:,:,:,tf.newaxis]
    
    @property
    def gamma_2_counts(self):
        return tf.transpose(self.train_sample[:,:,:,4], perm=[0,2,1])[:,:,:,tf.newaxis]
    
    @property
    def gamma_1_move(self):
        return self.move_args[:,0,:3]

    @property
    def gamma_1_rotate(self):
        return self.move_args[:,0,3:]
    
    @property
    def gamma_2_move(self):
        return self.move_args[:,1,:3]
    
    @property
    def gamma_2_rotate(self):
        return self.move_args[:,1,3:]
    
    @property
    def gamma_1_X_c(self):
        return tf.einsum('bcd,c->bd', tf.einsum('brcd->bcd', self.gamma_1_counts), SipmArray().sipm_center_x) / tf.einsum('brcd->bd', self.gamma_1_counts)
    
    @property
    def gamma_1_Y_c(self):
        return tf.einsum('brd,r->bd', tf.einsum('brcd->brd', self.gamma_1_counts), SipmArray().sipm_center_y) / tf.einsum('brcd->bd', self.gamma_1_counts)
    
    @property
    def gamma_1_Z_c(self):
        return tf.constant(tf.repeat(-3.25, self.train_sample.shape[0])[:, tf.newaxis].numpy(), dtype=tf.float64)
    
    @property
    def gamma_2_X_c(self):
        return tf.einsum('bcd,c->bd', tf.einsum('brcd->bcd', self.gamma_2_counts), SipmArray().sipm_center_x) / tf.einsum('brcd->bd', self.gamma_2_counts)
    
    @property
    def gamma_2_Y_c(self):
        return tf.einsum('brd,r->bd', tf.einsum('brcd->brd', self.gamma_2_counts), SipmArray().sipm_center_y) / tf.einsum('brcd->bd', self.gamma_2_counts)
    
    @property
    def gamma_2_Z_c(self):
        return tf.constant(tf.repeat(-3.25, self.train_sample.shape[0])[:, tf.newaxis].numpy(), dtype=tf.float64)
    
    @property
    def gamma_1_anger_local(self):
        return tf.transpose(tf.stack([self.gamma_1_X_c, self.gamma_1_Y_c, self.gamma_1_Z_c], axis=2)[:,0,:])
    
    @property
    def gamma_2_anger_local(self):
        return tf.transpose(tf.stack([self.gamma_2_X_c, self.gamma_2_Y_c, self.gamma_2_Z_c], axis=2)[:,0,:])
    
    @property
    def gamma_1_rotate_matrix(self):
        return tf.map_fn(rotate_matrix, self.gamma_1_rotate)
    
    @property
    def gamma_1_anger_global(self):
        return tf.einsum('bij,jb->bi', self.gamma_1_rotate_matrix, self.gamma_1_anger_local+tf.transpose(self.gamma_1_move))
    
    @property
    def gamma_2_rotate_matrix(self):
        return tf.map_fn(rotate_matrix, self.gamma_2_rotate)
    
    @property
    def gamma_2_anger_global(self):
        return tf.einsum('bij,jb->bi', self.gamma_2_rotate_matrix, self.gamma_2_anger_local+tf.transpose(self.gamma_2_move))
