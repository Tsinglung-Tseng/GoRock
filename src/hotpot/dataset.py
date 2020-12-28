from hotpot.simulation.sample import SampleWithAnger
from .functools import FuncArray


class AngerDataSet:
    def __init__(self, sample_df):
        self.sample_dfmple_df = sample_df
        self.swa = SampleWithAnger(sample_df)
        
    @property
    def sipm_counts_n_position(self):
        return tf.cast(
            FuncArray(self.swa.gamma_1_counts)
            .concatenate_with(
                FuncArray(self.swa.sipm_center_pos.shrink((1,2)).rollaxis(1,4).array[:,:,:,:3]), axis=3)
            .concatenate_with(
                FuncArray(tf.transpose(self.swa.gamma_2_counts, perm=[0,2,1,3])/10), axis=3)
            .concatenate_with(
                FuncArray(self.swa.sipm_center_pos.shrink((1,2)).rollaxis(1,4).array[:,:,:,3:]), axis=3)
            .to_tensor()
            , dtype=tf.float32
        )
    
    @property
    def anger_infered(self):
        return tf.cast(
            FuncArray(self.swa.gamma_1_anger_global)
            .concatenate_with(FuncArray(self.swa.gamma_2_anger_global), axis=1)
            .to_tensor()
            , dtype=tf.float32
        )
    
    @property
    def source_position(self):
        return tf.cast(
            FuncArray(self.swa.sample_df[['sourcePosX','sourcePosY','sourcePosZ']]).to_tensor(), 
            tf.float32
        )
